"""Functions for voxel grid creation and analysis."""

import numpy as np
from sklearn.mixture import GaussianMixture
import warnings

def create_voxel_grid(points, voxel_x_size, voxel_y_size, estimate_normals_func, 
                      calculate_tvu_func, verbose=False):
    """
    Create a voxel grid aligned with surface normal.
    
    Args:
        points (array): (N, 3) array of point coordinates
        voxel_x_size (float): Size of voxels in X dimension
        voxel_y_size (float): Size of voxels in Y dimension
        estimate_normals_func: Function to estimate normals
        calculate_tvu_func: Function to calculate TVU
        verbose (bool): Whether to output detailed information
            
    Returns:
        tuple: (voxel indices, x_size, y_size, z_size, rotation_matrix, 
               aligned points, average normal, mean point)
    """
    # Calculate average normal and create transformation matrix
    avg_normal, depth_mask = estimate_normals_func(points)
    
    # Create orthonormal basis with avg_normal as Z axis
    z_axis = avg_normal
    x_axis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(x_axis, z_axis)) > 0.9:
        x_axis = np.array([0.0, 1.0, 0.0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    
    # Create rotation matrix
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    
    # Get mean point for centering
    mean_point = np.mean(points, axis=0)
    
    # Transform points to align with normal
    points_centered = points - mean_point
    points_aligned = np.dot(points_centered, rotation_matrix)
    
    # Calculate voxel sizes in meters
    mean_depth = np.mean(points[:, 2])
    z_size = calculate_tvu_func(mean_depth)  # TVU for Z
    
    # Create grid indices
    x_idx = np.floor(points_aligned[:, 0] / voxel_x_size).astype(int)
    y_idx = np.floor(points_aligned[:, 1] / voxel_y_size).astype(int)
    z_idx = np.floor(points_aligned[:, 2] / z_size).astype(int)
    
    # Combine indices into single array
    voxel_indices = np.column_stack((x_idx, y_idx, z_idx))
    
    # Get unique voxels and counts
    unique_voxels, counts = np.unique(voxel_indices, axis=0, return_counts=True)
    
    if verbose:
        print(f"Created grid with {len(unique_voxels)} voxels")
        print(f"Voxel dimensions: X={voxel_x_size:.3f}m, Y={voxel_y_size:.3f}m, Z={z_size:.3f}m (TVU)")
    
    return voxel_indices, voxel_x_size, voxel_y_size, z_size, rotation_matrix, points_aligned, avg_normal, mean_point

def extract_all_modes(data, min_points_for_mode, max_modes, calculate_tvu_func):
    """
    Extract modes from depth values using Gaussian Mixture Model.
    
    Args:
        data (array): (N,) array of depth values
        min_points_for_mode (int): Minimum points needed for mode fitting
        max_modes (int): Maximum number of modes to fit
        calculate_tvu_func: Function to calculate TVU
        
    Returns:
        dict: Dictionary containing:
            - 'modes': List of dictionaries with 'mean' and 'std' for each mode
            - 'gmm': Fitted GMM object or None if not used
            - 'std': Standard deviation of the data
            - 'unimodal': Boolean indicating if data is unimodal
    """
    if len(data) == 0:
        return {'modes': [], 'gmm': None, 'std': 0.0, 'unimodal': True}
    
    # Calculate TVU for mean depth to use as threshold
    mean_depth = np.mean(data)
    tvu = calculate_tvu_func(mean_depth)
    max_std = tvu / 2  # Use half TVU as threshold
    
    # Calculate overall statistics
    data_std = np.std(data)
    data_mean = np.mean(data)
    
    # If too few points or std is small enough, return median
    if len(data) < min_points_for_mode or data_std <= max_std:
        return {
            'modes': [{'mean': data_mean, 'std': min(data_std, max_std)}],
            'gmm': None,
            'std': min(data_std, max_std),
            'unimodal': True
        }
    
    # Try fitting GMM with N modes
    n_modes = min(max_modes, len(data) // min_points_for_mode)
    
    try:
        # Initialize GMM
        gmm = GaussianMixture(n_components=n_modes,
                            random_state=42,
                            max_iter=100,
                            n_init=5)
        
        # Fit GMM
        gmm.fit(data.reshape(-1, 1))
        
        # Extract modes and sort by mean
        modes = []
        for mean, covar in zip(gmm.means_, gmm.covariances_):
            std = np.sqrt(covar.flatten()[0])
            # Clip standard deviation to max_std
            std = min(std, max_std)
            modes.append({'mean': mean[0], 'std': std})
        
        # Sort modes by mean depth
        modes.sort(key=lambda x: x['mean'])
        
        # Calculate probabilities for each mode using clipped standard deviations
        all_probs = np.zeros((len(data), len(modes)))
        for i, mode in enumerate(modes):
            # Calculate z-scores using clipped standard deviation
            z_scores = np.abs(data - mode['mean']) / mode['std']
            # Calculate probabilities using z-scores
            mode_probs = np.exp(-0.5 * z_scores**2)
            all_probs[:, i] = mode_probs
        
        # Update GMM with clipped standard deviations
        for i, mode in enumerate(modes):
            gmm.covariances_[i] = np.array([[mode['std']**2]])
        
        return {
            'modes': modes,
            'gmm': gmm,
            'std': min(data_std, max_std),
            'unimodal': n_modes == 1
        }
        
    except Exception as e:
        # If GMM fails, fall back to using median
        return {
            'modes': [{'mean': data_mean, 'std': min(data_std, max_std)}],
            'gmm': None,
            'std': min(data_std, max_std),
            'unimodal': True
        }