# ============================
# src/features/shot_maps.py
# ============================
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def make_grid(x_min=-25, x_max=25, y_min=-5, y_max=42, bin_size=1.0):
    """
    Create 2D bin edges in FEET for shot-chart coords.
    """
    xedges = np.arange(x_min, x_max + bin_size, bin_size, dtype=float)
    yedges = np.arange(y_min, y_max + bin_size, bin_size, dtype=float)
    return xedges, yedges


def make_player_maps(
    df_player: pd.DataFrame,
    xedges: np.ndarray,
    yedges: np.ndarray,
    value_col: str = "xPPS_offense",
    smooth_sigma: float = 1.25,
    eps: float = 1e-9,
):
    """
    Build per-player grids:
      density:  normalized attempt density over space
      quality:  smoothed E[value_col | location]
      impact:   density * quality
    Returns a dict with arrays and edges.
    """
    if value_col not in df_player.columns:
        raise ValueError(f"Missing value_col='{value_col}' in df_player.")

    if "SHOT_ATTEMPTED_FLAG" in df_player.columns:
        df_player = df_player[df_player["SHOT_ATTEMPTED_FLAG"] == 1]

    if "x_ft" in df_player.columns and "y_ft" in df_player.columns:
        x = df_player["x_ft"].to_numpy()
        y = df_player["y_ft"].to_numpy()
    elif "LOC_X" in df_player.columns and "LOC_Y" in df_player.columns:
        x = df_player["LOC_X"].to_numpy()
        y = df_player["LOC_Y"].to_numpy()
    else:
        raise ValueError("No x/y columns found. Expected (x_ft,y_ft) or (LOC_X,LOC_Y).")

    # counts and weighted sums
    H_cnt, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    w = df_player[value_col].to_numpy()
    H_sum, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=w)

    # smooth counts and sums, then ratio for stable quality
    H_cnt_s = gaussian_filter(H_cnt, sigma=smooth_sigma)
    H_sum_s = gaussian_filter(H_sum, sigma=smooth_sigma)
    quality = H_sum_s / (H_cnt_s + eps)

    # density normalized to sum=1
    density = H_cnt_s / (H_cnt_s.sum() + eps)

    impact = density * quality

    return {
        "attempt_count": int(df_player.shape[0]),
        "density": density.astype(np.float32),
        "quality": quality.astype(np.float32),
        "impact": impact.astype(np.float32),
        "xedges": np.asarray(xedges, dtype=np.float32),
        "yedges": np.asarray(yedges, dtype=np.float32),
    }


def build_player_maps(
    shots: pd.DataFrame,
    min_attempts: int = 200,
    value_col: str = "xPPS_offense",
    grid_kwargs: dict | None = None,
    smooth_sigma: float = 1.25,
):
    """
    Build maps for all players with >= min_attempts.
    Returns:
      maps: dict[PLAYER_ID] -> dict of arrays
      meta: pd.DataFrame with PLAYER_ID, PLAYER_NAME, attempts
    """
    if grid_kwargs is None:
        grid_kwargs = {}
    xedges, yedges = make_grid(**grid_kwargs)

    df = shots.copy()
    if "SHOT_ATTEMPTED_FLAG" in df.columns:
        df = df[df["SHOT_ATTEMPTED_FLAG"] == 1]

    counts = df.groupby("PLAYER_ID").size()
    eligible = counts[counts >= min_attempts].index.tolist()

    maps = {}
    rows = []
    for pid in eligible:
        dfp = df[df["PLAYER_ID"] == pid]
        pm = make_player_maps(
            dfp, xedges, yedges, value_col=value_col, smooth_sigma=smooth_sigma
        )
        maps[int(pid)] = pm
        pname = dfp["PLAYER_NAME"].iloc[0] if "PLAYER_NAME" in dfp.columns else str(pid)
        rows.append({"PLAYER_ID": int(pid), "PLAYER_NAME": pname, "attempts": pm["attempt_count"]})

    meta = pd.DataFrame(rows).sort_values("attempts", ascending=False).reset_index(drop=True)
    return maps, meta

def build_player_maps(
    shots: pd.DataFrame,
    min_attempts: int = 100,  # Lowered slightly
    value_col: str = "xPPS_offense",
    grid_kwargs: dict | None = None,
    smooth_sigma: float = 1.25,
):
    if grid_kwargs is None:
        grid_kwargs = {}
    xedges, yedges = make_grid(**grid_kwargs)

    df = shots.copy()
    if "SHOT_ATTEMPTED_FLAG" in df.columns:
        df = df[df["SHOT_ATTEMPTED_FLAG"] == 1]

    # 1. BUILD THE LEAGUE AVERAGE MAP FIRST
    league_map = make_player_maps(
        df, xedges, yedges, value_col=value_col, smooth_sigma=smooth_sigma
    )
    
    counts = df.groupby("PLAYER_ID").size()
    eligible = counts[counts >= min_attempts].index.tolist()

    maps = {}
    rows = []
    
    # 2. Add League Average as a special ID (e.g., ID 0)
    maps[0] = league_map
    rows.append({"PLAYER_ID": 0, "PLAYER_NAME": "LEAGUE_AVERAGE", "attempts": league_map["attempt_count"]})

    # 3. Build the rest of the eligible players
    for pid in eligible:
        dfp = df[df["PLAYER_ID"] == pid]
        pm = make_player_maps(
            dfp, xedges, yedges, value_col=value_col, smooth_sigma=smooth_sigma
        )
        maps[int(pid)] = pm
        pname = dfp["PLAYER_NAME"].iloc[0] if "PLAYER_NAME" in dfp.columns else str(pid)
        rows.append({"PLAYER_ID": int(pid), "PLAYER_NAME": pname, "attempts": pm["attempt_count"]})

    meta = pd.DataFrame(rows).sort_values("attempts", ascending=False).reset_index(drop=True)
    return maps, meta