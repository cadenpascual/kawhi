import numpy as np
import pandas as pd
import traceback
from src.tracking.release import find_release_frame_idx
from src.processing.indexing import attach_tracking_events_interval
from src.features.defense.helpers import (
    smooth_pos,
    central_vel,
    central_acc,
    central_diff,
    nan_stats_1d,
    get_player_xy,
    find_closest_defender_at_frame,
    extract_window_frames
)

# Feature calculation for a single shot, given its tracking event frames and release index

def compute_pre_shot_defense_features_refactored(
    event_frames,
    release_frame_idx,
    shooter_id,
    offense_team_id,
    *,
    fps=25,
    smooth_window=5,
    windows=(("w0", 0.0, 1.0), ("w1", 1.0, 3.0)),
    defender_mode="min_per_frame",
    include_accel=False,
    include_game_shot_clock=False,
    **kwargs
):
    """Calculates geometric features for a SINGLE shot."""
    if not event_frames:
        return {"error": "no_frames"}
    
    if release_frame_idx is None or release_frame_idx < 0 or release_frame_idx >= len(event_frames):
        return {"error": "bad_release_idx"}

    fr_rel = event_frames[release_frame_idx]
    
    # --- 1. Snapshot at Release ---
    shooter_xy_rel = get_player_xy(fr_rel, int(shooter_id))
    if shooter_xy_rel is None:
        return {"error": "shooter_missing_at_release"}
        
    did_rel, dxy_rel, dist_rel = find_closest_defender_at_frame(fr_rel, shooter_xy_rel, int(offense_team_id))
    defenders_on_court = len([p for p in fr_rel.get("players", []) if p.get("teamid") != int(offense_team_id)])

    out = {
        "release_frame_idx": int(release_frame_idx),
        "close_def_dist_release": float(dist_rel) if did_rel else np.nan,
        "closest_def_dist": float(dist_rel) if did_rel else np.nan, 
        "close_def_id": int(did_rel) if did_rel else -1,
        "num_defenders_tracked": int(defenders_on_court),
    }

    if include_game_shot_clock:
        out["game_clock_tracking"] = fr_rel.get("game_clock", np.nan)
        out["shot_clock_tracking"] = fr_rel.get("shot_clock", np.nan)

    dt = 1.0 / fps

    # --- 2. Window Analysis ---
    for wname, start_sec, end_sec in windows:
        idxs = extract_window_frames(event_frames, release_frame_idx, fps, start_sec, end_sec)
        T = len(idxs)
        if T < 5:
            out[f"{wname}_error"] = "too_few_frames"
            continue

        shooter_series = np.full((T, 2), np.nan)
        def_series = np.full((T, 2), np.nan)
        
        for t, idx in enumerate(idxs):
            fr = event_frames[idx]
            s_xy = get_player_xy(fr, int(shooter_id))
            if s_xy is not None:
                shooter_series[t] = s_xy
            
            if defender_mode == "release_defender" and did_rel:
                d_xy = get_player_xy(fr, did_rel)
                if d_xy is not None: def_series[t] = d_xy
            else: 
                cur_did, cur_dxy, _ = find_closest_defender_at_frame(fr, s_xy, int(offense_team_id))
                if cur_did: def_series[t] = cur_dxy

        s_smooth = smooth_pos(shooter_series, smooth_window)
        d_smooth = smooth_pos(def_series, smooth_window)
        dists = np.linalg.norm(s_smooth - d_smooth, axis=1)
        
        v_sh = central_vel(s_smooth, dt)
        v_df = central_vel(d_smooth, dt)
        speed_sh = np.linalg.norm(v_sh, axis=1)
        speed_df = np.linalg.norm(v_df, axis=1)

        dist_series = pd.Series(dists).interpolate().to_numpy()
        closing_speed = central_diff(dist_series, dt) 

        stats = nan_stats_1d(dists)
        s_stats = nan_stats_1d(speed_sh)
        d_stats = nan_stats_1d(speed_df)
        c_stats = nan_stats_1d(closing_speed)

        out.update({
            f"{wname}_close_def_dist_mean": stats["mean"],
            f"{wname}_close_def_dist_min": stats["min"],
            f"{wname}_shooter_speed_mean": s_stats["mean"],
            f"{wname}_shooter_speed_max": s_stats["max"],
            f"{wname}_def_speed_mean": d_stats["mean"],
            f"{wname}_closing_speed_mean": c_stats["mean"],
        })

        if include_accel:
            a_sh = np.linalg.norm(central_acc(v_sh, dt), axis=1)
            out[f"{wname}_shooter_accel_mean"] = nan_stats_1d(a_sh)["mean"]

    if "w0_shooter_speed_mean" in out:
        out["shooter_speed"] = out["w0_shooter_speed_mean"]

    return out


# Calculate features for all shots in a DataFrame, with robust error handling and optional metadata passthrough

def compute_defense_features_for_shots_refactored(
    shots_g,
    tracking_events, # List of dicts
    event_index,     # DataFrame
    *,
    fps=25,
    smooth_window=5,
    windows=(("w0", 0.0, 1.0), ("w1", 1.0, 3.0)),
    defender_mode="min_per_frame",
    include_accel=False,
    include_game_shot_clock=False,
    span_pad=4.0,
    max_time_diff=1.5,
    verbose_summary=True,
    # Updated default columns to pass through
    passthrough_cols=(
        "GAME_ID", 
        "SHOT_EVENT_ID", 
        "PERIOD", 
        "game_clock", 
        "PLAYER_ID", 
        "TEAM_ID", 
        "x_ft", 
        "y_ft", 
        "xFG_offense", 
        "xPPS_offense", 
        "SHOT_MADE_FLAG"
    )
):
    """
    Batch processor that computes defense features and includes requested metadata.
    """
    
    # --- Step 1: Attach Events (Vectorized & Safe) ---
    if "event_list_idx" not in shots_g.columns:
        if verbose_summary:
            print("Attaching tracking events to shots (Vectorized)...")
        shots_with_events = attach_tracking_events_interval(
            shots_g, 
            event_index, 
            span_pad=span_pad
        )
    else:
        shots_with_events = shots_g.copy()

    # --- Step 2: Iterate ---
    rows = []
    drop = {"no_event": 0, "no_release": 0, "feat_error": 0, "exception": 0, "ok": 0}

    for idx, shot in shots_with_events.iterrows():
        try:
            ev_idx = shot.get("event_list_idx")
            if pd.isna(ev_idx) or ev_idx == -1:
                drop["no_event"] += 1
                continue

            # Access List (Robustly)
            event = tracking_events[int(ev_idx)]
            frames = event.get("frames", [])
            
            if not frames:
                drop["no_event"] += 1
                continue

            # Find Release
            release_idx, _ = find_release_frame_idx(
                event_frames=frames,
                shot_game_clock=float(shot["game_clock"]),
                match="prev",
                max_time_diff=max_time_diff
            )
            
            if release_idx is None:
                drop["no_release"] += 1
                continue

            # Compute Features (Calls your separated helper function)
            feats = compute_pre_shot_defense_features_refactored(
                event_frames=frames,
                release_frame_idx=release_idx,
                shooter_id=int(shot["PLAYER_ID"]),
                offense_team_id=int(shot["TEAM_ID"]),
                fps=fps,
                smooth_window=smooth_window,
                windows=windows,
                defender_mode=defender_mode,
                include_accel=include_accel,
                include_game_shot_clock=include_game_shot_clock
            )

            if "error" in feats:
                drop["feat_error"] += 1
                continue

            # Tag row with the index
            feats["shot_index"] = idx
            feats["event_list_idx"] = int(ev_idx)
            feats["tracking_event_id"] = event.get("event_id", event.get("eventId"))

            rows.append(feats)
            drop["ok"] += 1

        except Exception as e:
            drop["exception"] += 1
            if drop["exception"] <= 1:
                print(f"Error on shot {idx}: {e}")
            continue

    if verbose_summary:
        print(f"Defense features summary: {drop}")
        
    if not rows:
        return pd.DataFrame()

    # --- Step 3: Join Passthrough Columns ---
    df_feats = pd.DataFrame(rows).set_index("shot_index")
    
    # Identify which requested columns actually exist in the input
    if passthrough_cols:
        available_cols = [c for c in passthrough_cols if c in shots_g.columns]
        if available_cols:
            # Join left on index to attach metadata
            df_feats = df_feats.join(shots_g[available_cols], how="left")

    # Reorder columns to have passthrough metadata first, followed by features
    desired_order = [
        'GAME_ID', 'SHOT_EVENT_ID', 'tracking_event_id', 'release_frame_idx', 
        'event_list_idx', 'PERIOD', 'game_clock', 'PLAYER_ID', 'TEAM_ID', 
        'x_ft', 'y_ft', 'xFG_offense', 'xPPS_offense', 'SHOT_MADE_FLAG', 
        'close_def_dist_release', 'closest_def_dist', 'close_def_id', 
        'num_defenders_tracked', 'w0_close_def_dist_mean', 'w0_close_def_dist_min',
        'w0_shooter_speed_mean', 'w0_shooter_speed_max', 'w0_def_speed_mean',
        'w0_closing_speed_mean', 'w1_close_def_dist_mean', 'w1_close_def_dist_min',
        'w1_shooter_speed_mean', 'w1_shooter_speed_max', 'w1_def_speed_mean', 
        'w1_closing_speed_mean', 'shooter_speed'
    ]

    existing_cols = [c for c in desired_order if c in df_feats.columns]
    remaining_cols = [c for c in df_feats.columns if c not in desired_order]
            
    return df_feats[existing_cols + remaining_cols]
