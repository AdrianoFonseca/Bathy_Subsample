"""Functions for estimating surface normals from point clouds."""

import numpy as np
from sklearn.neighbors import KDTree

def estimate_normals(points, k=20, calculate_tvu_func=None, verbose=False):
    """
    Estimate normal vectors using points within ±3*TVU of median depth.
    
    Args:
        points (array): (N, 3) array of point coordinates
        k (int): Number of nearest neighbors for normal estimation
        calculate_tvu_func: Function to calculate Total Vertical Uncertainty
        verbose (bool): Whether to output detailed information
        
    Returns:
        tuple: (average normal vector, depth mask)
    """
    # Get median depth and TVU
    median_depth = np.median(points[:, 2])
    
    if calculate_tvu_func:
        tvu = calculate_tvu_func(median_depth)
    else:
        # Default TVU calculation if none provided
        tvu = 0.15  # Default constant
    
    # Select points within ±3*TVU of median depth
    depth_mask = np.abs(points[:, 2] - median_depth) <= 3 * tvu
    points_for_normal = points[depth_mask]
    
    if verbose:
        print(f"Using {np.sum(depth_mask)} points within ±3*TVU of median depth ({median_depth:.2f}m) for normal estimation")
    
    # Ensure we have enough points
    k = min(k, len(points_for_normal) - 1)
    if k < 3:
        if verbose:
            print("Warning: Not enough points for normal estimation, defaulting to vertical")
        return np.array([0, 0, 1]), depth_mask
    
    # Build KD-tree for nearest neighbor search
    tree = KDTree(points_for_normal)
    
    # Find k nearest neighbors for each point
    distances, indices = tree.query(points_for_normal, k=k)
    
    # Compute normal for each point using PCA
    normals = []
    for i in range(len(points_for_normal)):
        neighbors = points_for_normal[indices[i]]
        centered = neighbors - np.mean(neighbors, axis=0)
        cov = np.dot(centered.T, centered)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        normal = eigenvecs[:, 0]  # Smallest eigenvector is normal
        # Ensure normal points "up" (positive Z)
        if normal[2] < 0:
            normal = -normal
        normals.append(normal)
    
    # Average the normals
    avg_normal = np.mean(normals, axis=0)
    avg_normal = avg_normal / np.linalg.norm(avg_normal)  # Normalize
    
    return avg_normal, depth_mask