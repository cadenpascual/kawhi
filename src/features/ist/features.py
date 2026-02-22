import numpy as np
import pandas as pd

# =====================================================================
# 1. MATH FUNCTIONS (Updated)
# =====================================================================

def sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def openness(
    dist: float,
    closing_speed: float,
    *,
    d0: float = 4.0,
    k_dist: float = 1.2,
    k_close: float = 0.6,
) -> float:
    """
    Openness score in (0,1).
    
    Parameters
    ----------
    dist : float
        Defender distance (feet).
    closing_speed : float
        Derivative of distance (feet/sec). 
        Negative = Closing in.
        Positive = Opening up (ignored/clamped to 0).
    """
    # 1. "Only take in negative values": 
    # If defender is running away (positive speed), treat impact as 0.
    valid_closing = min(0.0, closing_speed)
    
    # 2. Formula:
    # - Start with distance term
    # - Add closing term (since valid_closing is negative, this SUBTRACTS from z)
    z = k_dist * (dist - d0) + (k_close * valid_closing)
    
    return sigmoid(z)


def shootability(speed: float, accel: float, v0: float = 10.0, a0: float = 20.0) -> float:
    """Penalty for shooter movement."""
    return float(np.exp(-(speed / v0) ** 2 - (accel / a0) ** 2))


def sample_grid_nearest(grid, xedges, yedges, x, y) -> float:
    """Spatial lookup."""
    ix = np.searchsorted(xedges, x, side="right") - 1
    iy = np.searchsorted(yedges, y, side="right") - 1
    ix = np.clip(ix, 0, grid.shape[0] - 1)
    iy = np.clip(iy, 0, grid.shape[1] - 1)
    return float(grid[ix, iy])


def compute_ist_row(
    pid: int,
    x: float,
    y: float,
    dist: float,
    c_speed: float,
    s_speed: float,
    s_accel: float,
    maps_npz: dict,
    pid2row: dict,
    use: str
) -> dict:
    """Row-level calculation helper."""
    
    # 1. Q (Spatial Quality)
    if int(pid) in pid2row:
        i = pid2row[int(pid)]
        grid = maps_npz[use][i]
        Q = sample_grid_nearest(grid, maps_npz["xedges"], maps_npz["yedges"], x, y)
    else:
        Q = 0.45 # Fallback average

    # 2. O (Openness)
    O = openness(dist, c_speed)

    # 3. S (Shootability)
    S = shootability(s_speed, s_accel)

    # 4. IST
    IST = Q * O * S
    
    return {"IST": IST, "Q": Q, "O": O, "S": S}


# =====================================================================
# 2. MAIN BATCH FUNCTION (Hardcoded Columns)
# =====================================================================

def add_ist_column(df, maps, pid2row, use="quality"):
    """
    Adds IST, IST_Q, IST_O, IST_S columns using the specific defense feature columns.
    """
    out = df.copy()

    # Columns based on your provided schema
    col_dist0    = "w0_close_def_dist_mean"
    col_closing0 = "w0_closing_speed_mean"
    col_speed0   = "w0_shooter_speed_mean"

    col_dist1    = "w1_close_def_dist_mean"
    col_closing1 = "w1_closing_speed_mean"
    col_speed1   = "w1_shooter_speed_mean"
    
    # Accel might be missing if include_accel=False, handle gracefully
    col_accel  = "w0_shooter_accel_mean"
    
    # Prepare list for results
    ist_data = []

    # Pre-check columns exist to give clear error
    has_accel = col_accel in out.columns

    # Loop efficiently
    # We use explicit column lookups inside the loop for safety mixed with speed
    for row in out.itertuples(index=False):
        # Map values from named tuple
        # getattr is safe if column names have spaces, though yours don't
        pid = getattr(row, "PLAYER_ID")
        x   = getattr(row, "x_ft")
        y   = getattr(row, "y_ft")
        
        dist0    = getattr(row, col_dist0)
        closing0 = getattr(row, col_closing0)
        speed0   = getattr(row, col_speed0)

        dist1    = getattr(row, col_dist1)
        closing1 = getattr(row, col_closing1)
        speed1   = getattr(row, col_speed1)

        # Weighted combine (recency weighted example)
        dist    = 0.7 * dist0 + 0.3 * dist1
        c_speed = 0.7 * closing0 + 0.3 * closing1
        s_speed = 0.7 * speed0 + 0.3 * speed1
        
        s_accel = getattr(row, col_accel) if has_accel else 0.0

        res = compute_ist_row(
            pid=pid,
            x=float(x), 
            y=float(y),
            dist=float(dist), 
            c_speed=float(c_speed),
            s_speed=float(s_speed), 
            s_accel=float(s_accel),
            maps_npz=maps,
            pid2row=pid2row,
            use=use
        )
        ist_data.append(res)

    # Assign back to DataFrame
    temp_df = pd.DataFrame(ist_data, index=out.index)
    out["IST"]   = temp_df["IST"]
    out["IST_Q"] = temp_df["Q"]
    out["IST_O"] = temp_df["O"]
    out["IST_S"] = temp_df["S"]

    return out