import pandas as pd
import numpy as np
from src.features.shots.geometry import normalize_coordinates, filter_halfcourt, add_spatial_features

def test_pipeline_data_integrity():
    """Tests if the feature engineering functions handle dataframes correctly without dropping rows unexpectedly."""
    
    # 1. Create fake raw data
    dummy_data = pd.DataFrame({
        "LOC_X": [120, 0, -240], # 10ft, 0ft, -20ft
        "LOC_Y": [120, 50, 300], # 10ft, 4.1ft, 25ft
        "SHOT_TYPE": ["2PT Field Goal", "2PT Field Goal", "3PT Field Goal"]
    })
    
    # 2. Run it through your exact pipeline functions
    df = normalize_coordinates(dummy_data.copy())
    df = filter_halfcourt(df)
    df = add_spatial_features(df)
    
    # 3. Assert the outputs are what we expect
    assert "x_ft" in df.columns, "Failed to create x_ft"
    assert "shot_dist" in df.columns, "Failed to create shot_dist"
    assert df["is_three"].iloc[2] == 1, "Failed to flag 3-pointer correctly"
    
    # Make sure we didn't lose any rows (all are within halfcourt)
    assert len(df) == 3, "Rows were dropped incorrectly during filtering"