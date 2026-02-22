import numpy as np
import pandas as pd
import traceback


def rolling_center_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Centered rolling mean with reasonable min_periods."""
    if w is None or w <= 1:
        return x
    return (
        pd.Series(x)
        .rolling(window=w, center=True, min_periods=max(2, w // 2))
        .mean()
        .to_numpy()
    )

def smooth_pos(pos: np.ndarray, w: int) -> np.ndarray:
    """Smooth (T,2) positions coordinate-wise."""
    if w is None or w <= 1:
        return pos
    out = pos.copy()
    out[:, 0] = rolling_center_mean(out[:, 0], w)
    out[:, 1] = rolling_center_mean(out[:, 1], w)
    return out

def central_diff(arr: np.ndarray, dt: float) -> np.ndarray:
    """Central difference for 1D array."""
    out = np.full_like(arr, np.nan, dtype=float)
    if len(arr) > 2:
        out[1:-1] = (arr[2:] - arr[:-2]) / (2.0 * dt)
    return out

def central_vel(pos: np.ndarray, dt: float) -> np.ndarray:
    """Central difference velocity for (T,2) positions."""
    v = np.full_like(pos, np.nan, dtype=float)
    if len(pos) > 2:
        v[1:-1] = (pos[2:] - pos[:-2]) / (2.0 * dt)
    return v

def central_acc(v: np.ndarray, dt: float) -> np.ndarray:
    """Central difference acceleration for (T,2) velocity."""
    return central_vel(v, dt)

def nan_stats_1d(x: np.ndarray) -> dict:
    """Robust summary stats for 1D series, handling NaNs."""
    if x is None or len(x) == 0 or np.all(np.isnan(x)):
        return {k: np.nan for k in ["mean", "min", "max", "std", "p10", "p50", "p90"]}
    return {
        "mean": float(np.nanmean(x)),
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
        "std": float(np.nanstd(x)),
        "p10": float(np.nanpercentile(x, 10)),
        "p50": float(np.nanpercentile(x, 50)),
        "p90": float(np.nanpercentile(x, 90)),
    }

def get_player_xy(frame: dict, pid: int):
    """Return np.array([x,y]) or None."""
    for p in frame.get("players", []):
        if p.get("playerid") == pid:
            return np.array([p.get("x", np.nan), p.get("y", np.nan)], dtype=float)
    return None

def find_closest_defender_at_frame(frame: dict, shooter_xy: np.ndarray, offense_team_id: int):
    """Return (def_id, def_xy, dist) for closest defender at that frame."""
    best_id, best_xy, best_dist = None, None, np.inf
    
    for p in frame.get("players", []):
        if p.get("teamid") == offense_team_id:
            continue
            
        d_xy = np.array([p.get("x", np.nan), p.get("y", np.nan)], dtype=float)
        if np.any(np.isnan(d_xy)) or np.any(np.isnan(shooter_xy)):
            continue
            
        dist = np.linalg.norm(shooter_xy - d_xy)
        if dist < best_dist:
            best_dist = dist
            best_id = int(p.get("playerid"))
            best_xy = d_xy

    if best_id is None:
        return None, None, np.nan
    return best_id, best_xy, float(best_dist)

def extract_window_frames(event_frames, release_idx, fps, start_sec, end_sec):
    """Extract indices for a window [release - end_sec, release - start_sec]."""
    n_start = int(round(start_sec * fps))
    n_end = int(round(end_sec * fps))
    # Indices count BACKWARDS from release (0.0 is release, 2.0 is 2s prior)
    idx_start = max(0, release_idx - n_end)
    idx_end = max(0, release_idx - n_start)
    
    if idx_start > idx_end:
        return []
    return list(range(idx_start, idx_end + 1))

