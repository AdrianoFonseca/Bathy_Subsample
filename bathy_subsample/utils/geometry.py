"""Geometric utility functions for point cloud processing."""

import numpy as np

def calculate_tvu(depth, a=0.15, b=0.0075):
    """
    Calculate Total Vertical Uncertainty for a given depth.
    
    Args:
        depth (float): Depth value in meters
        a (float): Constant component of TVU formula
        b (float): Depth-dependent component of TVU formula
        
    Returns:
        float: Total Vertical Uncertainty value
    """
    return np.sqrt(a**2 + (b * depth)**2)