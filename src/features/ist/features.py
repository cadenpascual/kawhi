import numpy as np
import pandas as pd

# =====================================================================
# 1. MATH FUNCTIONS (Updated)
# =====================================================================
def openness(dist: float, d0: float = 4.8, k_dist: float = 0.1) -> float:
    # closing speed is removed entirely based on the optimization!
    z = k_dist * (dist - d0)
    return float(1.0 / (1.0 + np.exp(-z)))


def shootability(speed: float, accel: float, v0: float = 15.0, a0: float = 20.0) -> float:
    """Penalty for shooter movement."""
    return float(np.exp(-(speed / v0) ** 2 - (accel / a0) ** 2))


def sample_grid_nearest(grid, xedges, yedges, x, y) -> float:
    """Spatial lookup."""
    ix = np.searchsorted(xedges, x, side="right") - 1
    iy = np.searchsorted(yedges, y, side="right") - 1
    ix = np.clip(ix, 0, grid.shape[0] - 1)
    iy = np.clip(iy, 0, grid.shape[1] - 1)
    return float(grid[ix, iy])


# =====================================================================
# 1. ROW HELPER FUNCTION
# =====================================================================

def compute_ist_row(
    pid: int,
    x: float,
    y: float,
    dist: float,
    c_speed: float,
    maps_npz: dict,
    pid2row: dict,
    use: str
) -> dict:
    """Row-level calculation helper."""
    
    # 1. Q (Spatial Quality)
    if int(pid) in pid2row:
        i = pid2row[int(pid)]
        grid = maps_npz[use][i]
        Q_base = sample_grid_nearest(grid, maps_npz["xedges"], maps_npz["yedges"], x, y)
    else:
        Q = 0.45 # Fallback average

    O_base = openness(dist)
    
    Q_weighted = Q_base ** 2.1631
    O_weighted = O_base ** 1.0313

    # 4. Final Empirical IST
    # Shootability (S) is gone because the optimizer set its weight to 0.0
    IST = Q_weighted * O_weighted
    
    return {
        "Real_IST": IST, 
        "Base_Q": Q_base, 
        "Base_O": O_base
    }


# =====================================================================
# 2. MAIN BATCH FUNCTION
# =====================================================================

def add_ist_column(df, maps, pid2row, use="quality"):
    """
    Adds Real_IST, Real_Q, Real_O, Real_S columns using the specific w0 defense feature columns.
    """
    out = df.copy()

    # Columns based on your provided schema (strictly w0)
    col_dist    = "w0_close_def_dist_mean"
    col_closing = "w0_closing_speed_mean"
    col_speed   = "w0_shooter_speed_mean"
    
    # Accel might be missing if include_accel=False, handle gracefully
    col_accel  = "w0_shooter_accel_mean"
    
    # Prepare list for results
    ist_data = []

    # Pre-check columns exist to give clear error
    has_accel = col_accel in out.columns

    # Loop efficiently
    for row in out.itertuples(index=False):
        # Map values from named tuple
        pid = getattr(row, "PLAYER_ID")
        x   = getattr(row, "x_ft")
        y   = getattr(row, "y_ft")
        
        dist    = getattr(row, col_dist)
        closing = getattr(row, col_closing)
        speed   = getattr(row, col_speed)

        s_accel = getattr(row, col_accel) if has_accel else 0.0

        res = compute_ist_row(
            pid=pid,
            x=float(x), 
            y=float(y),
            dist=float(dist), 
            c_speed=float(closing),
            s_speed=float(speed), 
            s_accel=float(s_accel),
            maps_npz=maps,
            pid2row=pid2row,
            use=use
        )
        ist_data.append(res)

    # Assign back to DataFrame using the updated keys
    temp_df = pd.DataFrame(ist_data, index=out.index)
    out["Real_IST"] = temp_df["Real_IST"]
    out["Real_Q"]   = temp_df["Real_Q"]
    out["Real_O"]   = temp_df["Real_O"]
    out["Real_S"]   = temp_df["Real_S"]

    return out