import numpy as np

def normalize_coordinates(shots):
    """Converts raw API coordinates to feet."""
    shots["x_ft"] = shots["LOC_X"] / 12
    shots["y_ft"] = shots["LOC_Y"] / 12
    return shots

def filter_halfcourt(shots):
    """Removes shots taken beyond halfcourt."""
    shots = shots[(shots["y_ft"] >= -5) & (shots["y_ft"] <= 42)].copy()
    return shots.reset_index(drop=True)

def add_spatial_features(shots):
    """Calculates distances, angles, and zones."""
    shots["shot_dist"] = np.sqrt(shots["x_ft"]**2 + shots["y_ft"]**2)
    shots["angle"] = np.arctan2(shots["x_ft"], shots["y_ft"])
    
    # Shot Type indicators
    shots["is_three"] = shots["SHOT_TYPE"].str.contains("3PT").astype(int)
    shots["is_corner"] = ((np.abs(shots["x_ft"]) > 22) & (shots["y_ft"] < 14)).astype(int)
    
    return shots