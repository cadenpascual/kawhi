import numpy as np
from src.features.shots.geometry import calculate_shot_distance, is_corner_three

def test_calculate_shot_distance():
    """Test that the Euclidean distance formula works (3-4-5 triangle)."""
    # A shot taken 3 ft right and 4 ft up should be 5 ft away
    dist = calculate_shot_distance(3, 4)
    assert dist == 5.0, f"Expected 5.0, but got {dist}"

    # A shot at the rim (0,0) should be 0 ft away
    dist_rim = calculate_shot_distance(0, 0)
    assert dist_rim == 0.0, "Distance from rim should be 0"

def test_is_corner_three():
    """Test the geometric bounds of a corner three-pointer."""
    # True Corner 3: x < 14 and y >= 22
    assert is_corner_three(12.0, 23.0) == True, "Failed to identify valid corner 3"
    assert is_corner_three(12.0, -23.0) == True, "Failed to identify valid negative-y corner 3"

    # False Corner 3: Too far out (x > 14)
    assert is_corner_three(15.0, 23.0) == False, "Incorrectly flagged a non-corner 3"
    
    # False Corner 3: Too high up (y < 22)
    assert is_corner_three(10.0, 20.0) == False, "Incorrectly flagged a non-corner 3"