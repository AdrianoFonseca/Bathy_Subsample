"""Statistical functions for analyzing point cloud data."""

import numpy as np
from scipy.stats import norm as scipy_norm
from diptest import diptest

def summarize_statistics(original_points, processed_points, voxel_stats):
    """
    Create a summary of processing statistics.
    
    Args:
        original_points (array): Original point cloud data
        processed_points (array): Processed point cloud data
        voxel_stats (dict): Dictionary of voxel statistics
        
    Returns:
        dict: Dictionary of summary statistics
    """
    original_point_count = len(original_points)
    final_point_count = len(processed_points)
    reduction_percentage = ((original_point_count - final_point_count) / original_point_count) * 100
    
    stats = {
        'original_points': original_point_count,
        'processed_points': final_point_count,
        'points_reduced': original_point_count - final_point_count,
        'reduction_percentage': reduction_percentage,
    }
    
    # Add voxel statistics
    stats.update(voxel_stats)
    
    return stats