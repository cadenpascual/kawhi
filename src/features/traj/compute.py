import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from src.tracking.release import find_release_frame_idx
from src.processing.indexing import attach_tracking_events_interval
from src.features.defense.helpers import get_player_xy
from src.utils.casting import safe_int, safe_float
from src.features.ist.features import sample_grid_nearest
from src.tracking.release import locate_true_release_frame


# =====================================================================
# Geometric Transformations & Trajectory Extraction
# =====================================================================

def normalize_court_geometry(x_raw: float, y_raw: float, should_flip: bool) -> Tuple[float, float]:
    """Translates absolute coordinates to a relative, hoop-centric topological space."""
    if pd.isna(x_raw) or pd.isna(y_raw):
        return np.nan, np.nan
    x_f, y_f = (94.0 - x_raw, 50.0 - y_raw) if should_flip else (x_raw, y_raw)
    return (y_f - 25.0), (x_f - 5.25)


def extract_spatiotemporal_trajectories(
    event_frames: List[Dict],
    true_release_idx: int,
    pbp_frame_idx: int,               # <-- NEW: Needed to extend the window
    shooter_id: int,
    offense_team_id: int,
    maps_npz: Dict[str, Any],
    pid2row: Dict[int, int],
    fps: int = 25,
    time_window_sec: float = 3.0,     # Pre-shot window
    post_pbp_buffer_sec: float = 1.0  # <-- NEW: Buffer after the shot resolves
) -> Dict[str, Any]:
    """Extracts continuous kinematic trajectories for the pre-shot window and post-shot flight."""
    if not event_frames: return {"error": "no_frames"}
    
    rel_frame = event_frames[true_release_idx]
    shooter_xy_rel = get_player_xy(rel_frame, shooter_id)
    if shooter_xy_rel is None: return {"error": "shooter_missing"}
        
    should_flip = bool(shooter_xy_rel[0] > 47.0)

    # Define the new extended window
    start_idx = max(0, true_release_idx - int(time_window_sec * fps))
    end_idx = min(len(event_frames) - 1, pbp_frame_idx + int(post_pbp_buffer_sec * fps))
    temporal_window = event_frames[start_idx : end_idx + 1]
    
    initial_players = temporal_window[0].get("players", [])
    off_pids = [p['playerid'] for p in initial_players if p['teamid'] == offense_team_id]
    def_pids = [p['playerid'] for p in initial_players if p['teamid'] != offense_team_id]
    
    ordered_off = ([pid for pid in off_pids if pid == shooter_id] + [pid for pid in off_pids if pid != shooter_id])[:5]
    ordered_def = def_pids[:5]

    config = {
        "release_frame_global_idx": true_release_idx,
        "pbp_frame_global_idx": pbp_frame_idx,
        "local_release_idx": true_release_idx - start_idx,  # <-- CRITICAL for ML slicing
        "local_pbp_idx": pbp_frame_idx - start_idx,         # <-- CRITICAL for visualization
        "flipped_coordinates": int(should_flip),
        "ball_x_traj": [], "ball_y_traj": [], "ball_z_traj": [] # Added Z for visual flight arc
    }
    
    for i in range(1, 6):
        config.update({
            f"off{i}_pid": ordered_off[i-1] if i-1 < len(ordered_off) else np.nan,
            f"off{i}_x_traj": [], f"off{i}_y_traj": [], f"off{i}_q_traj": [],
            f"def{i}_pid": ordered_def[i-1] if i-1 < len(ordered_def) else np.nan,
            f"def{i}_x_traj": [], f"def{i}_y_traj": []
        })

    for frame in temporal_window:
        f_players = {p['playerid']: p for p in frame.get("players", [])}
        
        # Ball extraction (now capturing Z for true flight dynamics)
        b_data = frame.get("ball")
        bx, by, bz = np.nan, np.nan, np.nan
        if isinstance(b_data, dict):
            bx, by, bz = b_data.get("x", np.nan), b_data.get("y", np.nan), b_data.get("z", np.nan)
        elif isinstance(b_data, list) and len(b_data) >= 3:
            bx, by, bz = b_data[0], b_data[1], b_data[2]
            
        bx_norm, by_norm = normalize_court_geometry(bx, by, should_flip)
        config["ball_x_traj"].append(bx_norm)
        config["ball_y_traj"].append(by_norm)
        config["ball_z_traj"].append(bz)

        for i, pid in enumerate(ordered_off, start=1):
            p_data = f_players.get(pid, {})
            px, py = normalize_court_geometry(p_data.get('x'), p_data.get('y'), should_flip)
            config[f"off{i}_x_traj"].append(px)
            config[f"off{i}_y_traj"].append(py)
            
            q_val = 0.0
            pid_int = safe_int(pid, default=-1)
            lookup_id = pid_int if pid_int in pid2row else 0
            if lookup_id in pid2row and not np.isnan(px):
                q_val = sample_grid_nearest(maps_npz["quality"][pid2row[lookup_id]], maps_npz["xedges"], maps_npz["yedges"], px, py)
            config[f"off{i}_q_traj"].append(float(q_val))

        for i, pid in enumerate(ordered_def, start=1):
            p_data = f_players.get(pid, {})
            px, py = normalize_court_geometry(p_data.get('x'), p_data.get('y'), should_flip)
            config[f"def{i}_x_traj"].append(px)
            config[f"def{i}_y_traj"].append(py)

    return config

# =====================================================================
# Macro-Level Pipeline
# =====================================================================

def build_defensive_configurations(
    shots_g: pd.DataFrame,
    tracking_events: List[Dict],
    event_index: pd.DataFrame,
    maps_npz: Dict[str, Any],
    pid2row: Dict[int, int],
    show_diagnostics = False,
    fps: int = 25,
    span_pad: float = 4.0,
    max_time_diff: float = 1.5,
    passthrough_cols: Tuple[str, ...] = ("GAME_ID", "SHOT_EVENT_ID", "PERIOD", "game_clock", "PLAYER_ID", "TEAM_ID","SHOT_MADE_FLAG")
) -> pd.DataFrame:
    df_shots = attach_tracking_events_interval(shots_g, event_index, span_pad=span_pad) if "event_list_idx" not in shots_g.columns else shots_g.copy()

    records, diagnostics = [], {"no_event": 0, "no_release": 0, "feat_error": 0, "exception": 0, "processed": 0}

    for idx, shot in df_shots.iterrows():
        try:
            ev_idx = shot.get("event_list_idx")
            if pd.isna(ev_idx) or ev_idx == -1: diagnostics["no_event"] += 1; continue
            
            event = tracking_events[int(ev_idx)]
            frames = event.get("frames", [])
            if not frames: diagnostics["no_event"] += 1; continue

            pbp_release_idx, _ = find_release_frame_idx(frames, float(shot["game_clock"]), match="prev", max_time_diff=max_time_diff)
            if pbp_release_idx is None: diagnostics["no_release"] += 1; continue

            # --- NEW: Anchor the window to the true release, not the PBP log ---
            true_release_idx = locate_true_release_frame(frames, pbp_release_idx, fps=fps)

            config_state = extract_spatiotemporal_trajectories(
                event_frames=frames,
                true_release_idx=true_release_idx,
                pbp_frame_idx=pbp_release_idx,
                shooter_id=int(shot["PLAYER_ID"]),
                offense_team_id=int(shot["TEAM_ID"]),
                maps_npz=maps_npz, pid2row=pid2row, fps=fps
            )

            if "error" in config_state: diagnostics["feat_error"] += 1; continue

            config_state.update({"shot_index": idx, "event_list_idx": int(ev_idx), "tracking_event_id": event.get("event_id", event.get("eventId"))})
            records.append(config_state)
            diagnostics["processed"] += 1

        except Exception as e:
            diagnostics["exception"] += 1
            if diagnostics["exception"] == 1: print(f"Diagnostics Alert - Interruption at index {idx}: {e}")

    if show_diagnostics:
        print(f"Pipeline Diagnostics: {diagnostics}")
    if not records: return pd.DataFrame()

    df_features = pd.DataFrame(records).set_index("shot_index")
    if passthrough_cols:
        df_features = df_features.join(shots_g[[c for c in passthrough_cols if c in shots_g.columns]], how="left")

    metadata_cols = ['GAME_ID', 'SHOT_EVENT_ID', 'tracking_event_id', 'release_frame_idx', 'event_list_idx', 'PERIOD', 'game_clock', 'PLAYER_ID', 'TEAM_ID', 'flipped_coordinates', 'ball_x_traj', 'ball_y_traj']
    trajectory_cols = [col for i in range(1, 6) for col in (f"off{i}_pid", f"off{i}_x_traj", f"off{i}_y_traj", f"off{i}_q_traj", f"def{i}_pid", f"def{i}_x_traj", f"def{i}_y_traj")]

    ordered_schema = [c for c in (metadata_cols + trajectory_cols) if c in df_features.columns]
    return df_features[ordered_schema + [c for c in df_features.columns if c not in ordered_schema]]