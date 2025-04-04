from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import random as rn
import sys
import os
from sklearn.neighbors import KDTree
from collections import defaultdict
from tqdm import tqdm
from diptest import diptest
from sklearn.mixture import GaussianMixture
import warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import norm as scipy_norm
from joblib import Parallel, delayed
from matplotlib.colors import Normalize

# Add both the current directory and eif directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'eif'))

# Import version first
from version import VERSION, __version__
import eif as iso
print(f"Loading eif module from: {os.path.abspath(iso.__file__)}")
import seaborn as sb
sb.set_style(style="whitegrid")
sb.set_color_codes()

class IsolationGrid:
    """
    A class for processing point clouds using isolation forests and voxel grids.
    
    This class implements a method for denoising and subsampling point clouds by:
    1. Dividing the point cloud into leaves of nearest neighbors
    2. Using isolation forests to detect anomalies in each leaf
    3. Creating a voxel grid for each leaf
    4. Processing points within each voxel based on anomaly scores and mode fitting
    
    Attributes:
        group_size (int): Number of points to process in each leaf (default: 1000)
        voxel_x_size (float): Size of voxels in X dimension in meters (default: 1.0)
        voxel_y_size (float): Size of voxels in Y dimension in meters (default: 1.0)
        anomaly_threshold (float): Threshold for anomaly scores (default: 0.5)
        mode_probability_threshold (float): Minimum probability to assign point to a mode
        min_points_for_mode (int): Minimum points needed for mode fitting
        max_modes (int): Maximum number of modes to fit (default: 1)
        verbose (bool): Whether to output detailed processing information
        save_intermediate_files (bool): Whether to save intermediate files and visualizations
        plot_interval (int): Plot voxel distribution every N voxels (default: 500)
    """
    
    def __init__(self, group_size=1000, voxel_x_size=1.0, voxel_y_size=1.0, 
                 anomaly_threshold=0.5, mode_probability_threshold=0.3,
                 min_points_for_mode=3, max_modes=1, verbose=False, 
                 save_intermediate_files=False, plot_interval=500):
        """
        Initialize the IsolationGrid processor.
        
        Args:
            group_size (int): Number of points to process in each leaf
            voxel_x_size (float): Size of voxels in X dimension in meters
            voxel_y_size (float): Size of voxels in Y dimension in meters
            anomaly_threshold (float): Threshold for anomaly scores
            mode_probability_threshold (float): Minimum probability to assign point to a mode
            min_points_for_mode (int): Minimum points needed for mode fitting (default: 10)
            max_modes (int): Maximum number of modes to fit (default: 1)
            verbose (bool): Whether to output detailed processing information
            save_intermediate_files (bool): Whether to save intermediate files and visualizations
            plot_interval (int): Plot voxel distribution every N voxels (default: 500)
        """
        self.group_size = group_size
        self.voxel_x_size = voxel_x_size
        self.voxel_y_size = voxel_y_size
        self.anomaly_threshold = anomaly_threshold
        self.prob_threshold = mode_probability_threshold
        self.min_points_for_mode = min_points_for_mode  
        self.max_modes = max(1, max_modes)  # Ensure at least 1 mode
        self.verbose = verbose
        self.save_intermediate_files = save_intermediate_files
        self.plot_interval = plot_interval
        self.total_voxels_processed = 0  # Track total voxels across all leaves
        
    def calculate_tvu(self, depth, a=0.15, b=0.0075):
        """Calculate Total Vertical Uncertainty for a given depth."""
        return np.sqrt(a**2 + (b * depth)**2)
    
    def estimate_normals(self, points, k=20):
        """
        Estimate normal vectors using points within ±3*TVU of median depth.
        
        Args:
            points (array): (N, 3) array of point coordinates
            k (int): Number of nearest neighbors for normal estimation
            
        Returns:
            tuple: (average normal vector, depth mask)
        """
        # Get median depth and TVU
        median_depth = np.median(points[:, 2])
        tvu = self.calculate_tvu(median_depth)
        
        # Select points within ±3*TVU of median depth
        depth_mask = np.abs(points[:, 2] - median_depth) <= 3 * tvu
        points_for_normal = points[depth_mask]
        
        if self.verbose:
            print(f"Using {np.sum(depth_mask)} points within ±3*TVU of median depth ({median_depth:.2f}m) for normal estimation")
        
        # Ensure we have enough points
        k = min(k, len(points_for_normal) - 1)
        if k < 3:
            if self.verbose:
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
    
    def create_voxel_grid(self, points):
        """
        Create a voxel grid aligned with surface normal.
        
        Args:
            points (array): (N, 3) array of point coordinates
            
        Returns:
            tuple: (voxel indices, x_size, y_size, z_size, rotation_matrix, 
                   aligned points, average normal, mean point)
        """
        # Calculate average normal and create transformation matrix
        avg_normal, depth_mask = self.estimate_normals(points)
        
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
        z_size = self.calculate_tvu(mean_depth)  # TVU for Z
        
        # Create grid indices
        x_idx = np.floor(points_aligned[:, 0] / self.voxel_x_size).astype(int)
        y_idx = np.floor(points_aligned[:, 1] / self.voxel_y_size).astype(int)
        z_idx = np.floor(points_aligned[:, 2] / z_size).astype(int)
        
        # Combine indices into single array
        voxel_indices = np.column_stack((x_idx, y_idx, z_idx))
        
        # Get unique voxels and counts
        unique_voxels, counts = np.unique(voxel_indices, axis=0, return_counts=True)
        
        if self.verbose:
            print(f"Created grid with {len(unique_voxels)} voxels")
            print(f"Voxel dimensions: X={self.voxel_x_size:.3f}m, Y={self.voxel_y_size:.3f}m, Z={z_size:.3f}m (TVU)")
        
        return voxel_indices, self.voxel_x_size, self.voxel_y_size, z_size, rotation_matrix, points_aligned, avg_normal, mean_point
    
    def visualize_points(self, points, scores=None, title="Point Cloud", 
                        output_file=None, s=1, alpha=1.0):
        """
        Visualize points in 3D with optional coloring by scores.
        
        Args:
            points (array): (N, 3) array of point coordinates
            scores (array, optional): Values to use for coloring points
            title (str): Plot title
            output_file (str, optional): Path to save visualization
            s (float): Point size
            alpha (float): Point transparency
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if scores is not None:
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c=scores, cmap='viridis', s=s, alpha=alpha)
            plt.colorbar(scatter, label='Anomaly Score')
        else:
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c='blue', s=s, alpha=alpha)
        
        ax.set_title(title)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def extract_all_modes(self, data):
        """
        Extract modes from depth values using Gaussian Mixture Model.
        
        Args:
            data (array): (N,) array of depth values
            
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
        tvu = self.calculate_tvu(mean_depth)
        max_std = tvu / 2  # Use half TVU as threshold
        
        # Calculate overall statistics
        data_std = np.std(data)
        data_mean = np.mean(data)
        
        # If too few points or std is small enough, return median
        if len(data) < self.min_points_for_mode or data_std <= max_std:
            return {
                'modes': [{'mean': data_mean, 'std': min(data_std, max_std)}],
                'gmm': None,
                'std': min(data_std, max_std),
                'unimodal': True
            }
        
        # Try fitting GMM with N modes
        n_modes = min(self.max_modes, len(data) // self.min_points_for_mode)
        
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
            if self.verbose:
                print(f"GMM fitting failed: {str(e)}")
            # If GMM fails, fall back to using median
            return {
                'modes': [{'mean': data_mean, 'std': min(data_std, max_std)}],
                'gmm': None,
                'std': min(data_std, max_std),
                'unimodal': True
            }

    def visualize_leaf(self, points, leaf_id, output_dir, selected_points, high_anomaly_points, mode_points, low_prob_points):
        """
        Visualize a leaf of points with three subplots:
        1. Original point cloud
        2. Selected points that will be kept (colored by their selection criteria)
        3. Point depths
        
        Args:
            points (array): (N, 3) array of original point coordinates
            leaf_id (int): Identifier for the current leaf
            output_dir (str): Directory to save visualization
            selected_points (array): Points selected from all voxels
            high_anomaly_points (array): Points selected as high anomalies
            mode_points (array): Points selected as mode representatives
            low_prob_points (array): Points selected as low probability/outside std points
        """
        fig = plt.figure(figsize=(20, 6))
        
        # Original point cloud subplot
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c='blue', s=5, alpha=0.7)
        ax1.set_title(f'Leaf {leaf_id} - Original Point Cloud\n{len(points):,} points')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        
        # Selected points subplot - show points that will be kept
        ax2 = fig.add_subplot(132, projection='3d')
        
        # Plot high anomaly points
        if len(high_anomaly_points) > 0:
            ax2.scatter(high_anomaly_points[:, 0], high_anomaly_points[:, 1], high_anomaly_points[:, 2],
                      c='red', s=5, alpha=0.7, label='High Anomaly')
            
        # Plot mode points
        if len(mode_points) > 0:
            ax2.scatter(mode_points[:, 0], mode_points[:, 1], mode_points[:, 2],
                      c='blue', s=5, label='Mode/Median')
            
        # Plot low probability/outside std points
        if len(low_prob_points) > 0:
            ax2.scatter(low_prob_points[:, 0], low_prob_points[:, 1], low_prob_points[:, 2],
                      c='green', s=5, alpha=0.7, label='Low Prob/Outside')
        
        title_str = f'Leaf {leaf_id} - Selected Points\n'
        title_str += f'High Anomaly: {len(high_anomaly_points):,}\n'
        title_str += f'Mode/Median: {len(mode_points):,}\n'
        title_str += f'Low Prob/Outside: {len(low_prob_points):,}'
        
        ax2.set_title(title_str)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        ax2.legend()
        
        # Selection criteria subplot - show all points colored by their Z value
        ax3 = fig.add_subplot(133, projection='3d')
        scatter = ax3.scatter(points[:, 0], points[:, 1], points[:, 2],
                            c=points[:, 2], cmap='viridis', s=5, alpha=0.7)
        plt.colorbar(scatter, label='Z (m)')
        
        ax3.set_title(f'Leaf {leaf_id} - Point Depths')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Z (m)')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{output_dir}/leaf_{leaf_id}_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_group_with_iforest(self, points, leaf_id, output_dir):
        """
        Analyze a group of points with isolation forest and save results.
        For each voxel:
        - If N=1:
            - If too few points: use median/std filtering
            - If enough points: use unimodal GMM
        - If N>1:
            - If too few points: fall back to N=1 case
            - If enough points: use N-mode GMM
            
        Args:
            points (array): (N, 3) array of point coordinates
            leaf_id (int): Identifier for the current leaf
            output_dir (str): Directory to save results
            
        Returns:
            tuple: (processed points in original coordinate system, voxel statistics dictionary)
        """
        # Save arrays for leaf 0
        if leaf_id == 0:
            np.savetxt(f"{output_dir}/leaf_0_input.xyz", points)
        
        # Create isolation forest with normalized points
        points_for_iforest = points.copy()
        points_mean = np.mean(points_for_iforest, axis=0)
        points_std = np.std(points_for_iforest, axis=0)
        points_for_iforest = (points_for_iforest - points_mean) / points_std
        
        sample_size = min(256, len(points))
        iforest = iso.iForest(points_for_iforest, ntrees=200, sample_size=sample_size)
        scores = iforest.compute_paths(X_in=points_for_iforest)
        
        # Normalize scores to [0,1] range
        scores_normalized = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        
        # Create voxel grid and get transformation matrices
        voxel_indices, x_size, y_size, z_size, rotation_matrix, points_aligned, avg_normal, mean_point = self.create_voxel_grid(points)
        
        # Group points by voxel
        voxel_dict = defaultdict(list)
        for i, (point, score) in enumerate(zip(points_aligned, scores_normalized)):
            voxel_key = tuple(voxel_indices[i])
            voxel_dict[voxel_key].append((point, score, i))  # Store original index
        
        # Initialize lists for different point categories
        selected_points = []
        high_anomaly_points = []
        mode_points = []
        low_prob_points = []  # New list for low prob/outside std points
        
        # Initialize counters
        n_high_anomaly = 0
        n_low_anomaly = 0
        n_multimodal_points = 0
        n_extracted_modes = 0
        n_testable_voxels = 0
        
        # Track voxel categories
        n_voxels_too_few_points = 0
        n_voxels_unimodal = 0
        n_voxels_multimodal = 0
        total_voxels = len(voxel_dict)
        
        # Process each voxel
        for voxel_key, voxel_points in voxel_dict.items():
            self.total_voxels_processed += 1
            
            # Unpack points, scores, and indices
            voxel_points_array = np.array([p[0] for p in voxel_points])
            voxel_scores = np.array([p[1] for p in voxel_points])
            original_indices = [p[2] for p in voxel_points]
            
            # Keep high anomaly points
            high_anomaly_mask = voxel_scores > self.anomaly_threshold
            high_anomaly_points_voxel = voxel_points_array[high_anomaly_mask]
            if len(high_anomaly_points_voxel) > 0:
                high_anomaly_original = np.dot(high_anomaly_points_voxel, rotation_matrix.T) + mean_point
                high_anomaly_points.extend(high_anomaly_original)
                selected_points.extend(high_anomaly_original)
            n_high_anomaly += len(high_anomaly_points_voxel)
            
            # Get low anomaly points
            low_anomaly_mask = voxel_scores <= self.anomaly_threshold
            low_anomaly_points = voxel_points_array[low_anomaly_mask]
            
            # Skip if no low anomaly points
            if len(low_anomaly_points) == 0:
                continue
            
            # Process low anomaly points
            n_testable_voxels += 1
            
            # Extract modes from depth values
            result = self.extract_all_modes(low_anomaly_points[:, 2])
            
            # Skip if no modes found
            if len(result['modes']) == 0:
                continue
            
            # Track voxel category
            if len(low_anomaly_points) < self.min_points_for_mode:
                n_voxels_too_few_points += 1
            elif result['unimodal']:
                n_voxels_unimodal += 1
            else:
                n_voxels_multimodal += 1
            
            # Calculate TVU for this voxel's mean depth
            mean_depth = np.mean(low_anomaly_points[:, 2])
            tvu = self.calculate_tvu(mean_depth)
            std_threshold = tvu / 2  # Use half TVU as threshold
            
            # Process points based on number of points and modes
            if len(low_anomaly_points) < self.min_points_for_mode:
                # Too few points case - use median/std filtering
                if result['std'] <= std_threshold:
                    # Small spread - keep median point
                    median = result['modes'][0]['mean']
                    median_point = min(low_anomaly_points, key=lambda p: abs(p[2] - median))
                    median_point_original = np.dot(median_point.reshape(1, -1), rotation_matrix.T) + mean_point
                    mode_points.append(median_point_original[0])
                    selected_points.append(median_point_original[0])
                    n_extracted_modes += 1
                else:
                    # Large spread - keep median and outside points
                    median = result['modes'][0]['mean']
                    median_point = min(low_anomaly_points, key=lambda p: abs(p[2] - median))
                    median_point_original = np.dot(median_point.reshape(1, -1), rotation_matrix.T) + mean_point
                    mode_points.append(median_point_original[0])
                    selected_points.append(median_point_original[0])
                    n_extracted_modes += 1
                    
                    # Keep points outside std limit
                    outside_points = low_anomaly_points[abs(low_anomaly_points[:, 2] - median) > std_threshold]
                    if len(outside_points) > 0:
                        outside_points_original = np.dot(outside_points, rotation_matrix.T) + mean_point
                        low_prob_points.extend(outside_points_original)  # Track these points
                        selected_points.extend(outside_points_original)
            
            elif self.max_modes == 1:
                # N=1 case - use unimodal GMM
                if result['gmm'] is not None:
                    # GMM converged - keep points with probability < threshold
                    gmm = result['gmm']
                    probs = gmm.predict_proba(low_anomaly_points[:, 2].reshape(-1, 1))
                    keep_mask = probs[:, 0] < self.prob_threshold
                    
                    # Keep points with low probability
                    low_prob_points_subset = low_anomaly_points[keep_mask]
                    if len(low_prob_points_subset) > 0:
                        low_prob_points_original = np.dot(low_prob_points_subset, rotation_matrix.T) + mean_point
                        low_prob_points.extend(low_prob_points_original)  # Track these points
                        selected_points.extend(low_prob_points_original)
                    
                    # Keep one point from the mode
                    mode_mean = result['modes'][0]['mean']
                    mode_point = min(low_anomaly_points[~keep_mask], key=lambda p: abs(p[2] - mode_mean))
                    mode_point_original = np.dot(mode_point.reshape(1, -1), rotation_matrix.T) + mean_point
                    mode_points.append(mode_point_original[0])
                    selected_points.append(mode_point_original[0])
                    n_extracted_modes += 1
                else:
                    # GMM didn't converge - fall back to median/std filtering
                    median = result['modes'][0]['mean']
                    median_point = min(low_anomaly_points, key=lambda p: abs(p[2] - median))
                    median_point_original = np.dot(median_point.reshape(1, -1), rotation_matrix.T) + mean_point
                    mode_points.append(median_point_original[0])
                    selected_points.append(median_point_original[0])
                    n_extracted_modes += 1
                    
                    # Keep points outside std limit
                    outside_points = low_anomaly_points[abs(low_anomaly_points[:, 2] - median) > std_threshold]
                    if len(outside_points) > 0:
                        outside_points_original = np.dot(outside_points, rotation_matrix.T) + mean_point
                        low_prob_points.extend(outside_points_original)  # Track these points
                        selected_points.extend(outside_points_original)
            
            else:  # N > 1
                if len(low_anomaly_points) < self.max_modes * self.min_points_for_mode:
                    # Not enough points for N modes - fall back to N=1 case
                    if result['gmm'] is not None:
                        # GMM converged - keep points with probability < threshold
                        gmm = result['gmm']
                        probs = gmm.predict_proba(low_anomaly_points[:, 2].reshape(-1, 1))
                        keep_mask = probs[:, 0] < self.prob_threshold
                        
                        # Keep points with low probability
                        low_prob_points_subset = low_anomaly_points[keep_mask]
                        if len(low_prob_points_subset) > 0:
                            low_prob_points_original = np.dot(low_prob_points_subset, rotation_matrix.T) + mean_point
                            low_prob_points.extend(low_prob_points_original)  # Track these points
                            selected_points.extend(low_prob_points_original)
                        
                        # Keep one point from the mode
                        mode_mean = result['modes'][0]['mean']
                        mode_point = min(low_anomaly_points[~keep_mask], key=lambda p: abs(p[2] - mode_mean))
                        mode_point_original = np.dot(mode_point.reshape(1, -1), rotation_matrix.T) + mean_point
                        mode_points.append(mode_point_original[0])
                        selected_points.append(mode_point_original[0])
                        n_extracted_modes += 1
                    else:
                        # GMM didn't converge - fall back to median/std filtering
                        median = result['modes'][0]['mean']
                        median_point = min(low_anomaly_points, key=lambda p: abs(p[2] - median))
                        median_point_original = np.dot(median_point.reshape(1, -1), rotation_matrix.T) + mean_point
                        mode_points.append(median_point_original[0])
                        selected_points.append(median_point_original[0])
                        n_extracted_modes += 1
                        
                        # Keep points outside std limit
                        outside_points = low_anomaly_points[abs(low_anomaly_points[:, 2] - median) > std_threshold]
                        if len(outside_points) > 0:
                            outside_points_original = np.dot(outside_points, rotation_matrix.T) + mean_point
                            low_prob_points.extend(outside_points_original)  # Track these points
                            selected_points.extend(outside_points_original)
                else:
                    # Enough points for N modes - use N-mode GMM
                    if result['gmm'] is not None:
                        # GMM converged - process each mode
                        gmm = result['gmm']
                        probs = gmm.predict_proba(low_anomaly_points[:, 2].reshape(-1, 1))
                        
                        # Find points with low probability for ALL modes
                        keep_mask = np.all(probs < self.prob_threshold, axis=1)
                        low_prob_points_subset = low_anomaly_points[keep_mask]
                        if len(low_prob_points_subset) > 0:
                            low_prob_points_original = np.dot(low_prob_points_subset, rotation_matrix.T) + mean_point
                            low_prob_points.extend(low_prob_points_original)  # Track these points
                            selected_points.extend(low_prob_points_original)
                        
                        # Keep one point from each mode up to max_modes
                        for mode_idx in range(min(self.max_modes, probs.shape[1])):
                            # Find points assigned to this mode (highest probability)
                            mode_mask = np.argmax(probs, axis=1) == mode_idx
                            mode_points_subset = low_anomaly_points[mode_mask]
                            
                            if len(mode_points_subset) > 0:
                                # Keep one point closest to the mode mean
                                mode_mean = result['modes'][mode_idx]['mean']
                                mode_point = min(mode_points_subset, key=lambda p: abs(p[2] - mode_mean))
                                mode_point_original = np.dot(mode_point.reshape(1, -1), rotation_matrix.T) + mean_point
                                mode_points.append(mode_point_original[0])
                                selected_points.append(mode_point_original[0])
                                n_extracted_modes += 1
                    else:
                        # GMM didn't converge - fall back to median/std filtering
                        median = result['modes'][0]['mean']
                        median_point = min(low_anomaly_points, key=lambda p: abs(p[2] - median))
                        median_point_original = np.dot(median_point.reshape(1, -1), rotation_matrix.T) + mean_point
                        mode_points.append(median_point_original[0])
                        selected_points.append(median_point_original[0])
                        n_extracted_modes += 1
                        
                        # Keep points outside std limit
                        outside_points = low_anomaly_points[abs(low_anomaly_points[:, 2] - median) > std_threshold]
                        if len(outside_points) > 0:
                            outside_points_original = np.dot(outside_points, rotation_matrix.T) + mean_point
                            low_prob_points.extend(outside_points_original)  # Track these points
                            selected_points.extend(outside_points_original)
        
        # Convert lists to numpy arrays
        selected_points = np.array(selected_points)
        high_anomaly_points = np.array(high_anomaly_points) if high_anomaly_points else np.empty((0, 3))
        mode_points = np.array(mode_points) if mode_points else np.empty((0, 3))
        low_prob_points = np.array(low_prob_points) if low_prob_points else np.empty((0, 3))
        
        # Verify that our point categories sum up to selected points
        if self.verbose:
            # Sort all arrays to ensure consistent comparison
            selected_sorted = np.array(sorted(selected_points.tolist()))
            combined_sorted = np.array(sorted(np.vstack([
                high_anomaly_points,
                mode_points,
                low_prob_points
            ]).tolist() if len(high_anomaly_points) + len(mode_points) + len(low_prob_points) > 0 else []))
            
            if len(selected_sorted) != len(combined_sorted):
                print(f"\nWarning: Point count mismatch in leaf {leaf_id}!")
                print(f"Selected points: {len(selected_sorted)}")
                print(f"Sum of categories: {len(combined_sorted)}")
                print(f"  High anomaly: {len(high_anomaly_points)}")
                print(f"  Mode points: {len(mode_points)}")
                print(f"  Low prob/outside: {len(low_prob_points)}")
            elif len(selected_sorted) > 0 and not np.allclose(selected_sorted, combined_sorted, rtol=1e-10, atol=1e-10):
                print(f"\nWarning: Point content mismatch in leaf {leaf_id}!")
                print("Some points are different between selected and categorized points")
        
        # Save arrays for leaf 0 (for debugging/validation only)
        if leaf_id == 0 and self.verbose:
            np.savetxt(f"{output_dir}/leaf_0_input.xyz", points)
            np.savetxt(f"{output_dir}/leaf_0_output.xyz", selected_points)
        
        # Visualize the leaf if saving intermediates
        if self.save_intermediate_files:
            self.visualize_leaf(points, leaf_id, output_dir, selected_points, high_anomaly_points, mode_points, low_prob_points)
        
        # Return both points and voxel statistics
        voxel_stats = {
            'total_voxels': total_voxels,
            'too_few_points': n_voxels_too_few_points,
            'unimodal': n_voxels_unimodal,
            'multimodal': n_voxels_multimodal,
            'high_anomaly_points': n_high_anomaly,
            'low_prob_points': len(low_prob_points),
            'mode_points': len(mode_points)
        }
        
        return selected_points, voxel_stats

    def process_leaf(self, leaf_data):
        """
        Process a single leaf of points.
        
        Args:
            leaf_data (tuple): (points, leaf_id, output_dir)
            
        Returns:
            tuple: (processed points, voxel statistics)
        """
        points, leaf_id, output_dir = leaf_data
        return self.analyze_group_with_iforest(points, leaf_id, output_dir)

    def process(self, input_file, output_dir="isolation_forest_results", n_jobs=-1):
        """
        Process a point cloud file using the isolation grid method.
        
        Args:
            input_file (str): Path to input XYZ file
            output_dir (str): Directory to save results
            n_jobs (int): Number of parallel jobs. -1 means use all processors.
            
        Returns:
            tuple: (processed points array, processing statistics dict)
        """
        # Reset total voxel count at start of processing
        self.total_voxels_processed = 0

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load point cloud
        
        print("Loading point cloud...")
        points = np.loadtxt(input_file)
        original_point_count = len(points)
        
        print(f"Original point cloud has {original_point_count:,} points")
        print(f"\nCreating groups of {self.group_size:,} nearest neighbors...")
        
        # Create groups of nearest neighbors
        n_points = len(points)
        points_remaining = np.ones(n_points, dtype=bool)
        leaf_id = 0
        
        # Lists to store leaf data
        leaf_data = []
        points_processed = 0
        
        # Prepare leaves for processing
        while np.sum(points_remaining) >= self.group_size:
            unassigned_indices = np.where(points_remaining)[0]
            unassigned_points = points[unassigned_indices]
            
            tree = KDTree(unassigned_points)
            center_point = unassigned_points[0:1]
            
            _, neighbor_indices = tree.query(
                center_point, 
                k=self.group_size,
                return_distance=True,
                dualtree=True
            )
            neighbor_indices = neighbor_indices[0]
            
            selected_indices = unassigned_indices[neighbor_indices]
            group_points = points[selected_indices]
            points_processed += len(group_points)
            
            points_remaining[selected_indices] = False
            
            leaf_data.append((group_points, leaf_id, output_dir))
            leaf_id += 1
        
        # Process remaining points as final leaf
        remaining_indices = np.where(points_remaining)[0]
        if len(remaining_indices) > 0:
            remaining_points = points[remaining_indices]
            points_processed += len(remaining_points)
            leaf_data.append((remaining_points, leaf_id, output_dir))
        
        # Process all leaves in parallel with progress bar
        print("\nProcessing leaves...")
        with tqdm(total=len(leaf_data), desc="Processing leaves", disable=not self.verbose, position=0, leave=True) as pbar:
            def process_with_progress(leaf):
                result = self.process_leaf(leaf)
                pbar.update(1)
                return result
            
            all_processed_results = Parallel(n_jobs=n_jobs)(
                delayed(process_with_progress)(leaf) for leaf in leaf_data
            )
        
        if self.verbose:
            print()  # Add newline after progress bar
        
        # Separate points and statistics
        all_processed_points = [result[0] for result in all_processed_results]
        all_voxel_stats = [result[1] for result in all_processed_results]
        
        # Combine results
        final_points = np.vstack(all_processed_points)
        np.savetxt(f"{output_dir}/processed_points.xyz", final_points)
        
        # Combine voxel statistics
        total_voxels = sum(stats['total_voxels'] for stats in all_voxel_stats)
        total_too_few = sum(stats['too_few_points'] for stats in all_voxel_stats)
        total_unimodal = sum(stats['unimodal'] for stats in all_voxel_stats)
        total_multimodal = sum(stats['multimodal'] for stats in all_voxel_stats)
        total_high_anomaly = sum(stats['high_anomaly_points'] for stats in all_voxel_stats)
        total_low_prob = sum(stats.get('low_prob_points', 0) for stats in all_voxel_stats)
        
        # Calculate statistics
        final_point_count = len(final_points)
        reduction_percentage = ((original_point_count - final_point_count) / original_point_count) * 100
        
        stats = {
            'original_points': original_point_count,
            'processed_points': final_point_count,
            'points_reduced': original_point_count - final_point_count,
            'reduction_percentage': reduction_percentage,
            'points_processed': points_processed,
            'total_voxels': total_voxels,
            'voxels_too_few_points': total_too_few,
            'voxels_unimodal': total_unimodal,
            'voxels_multimodal': total_multimodal,
            'high_anomaly_points': total_high_anomaly,
            'low_prob_points': total_low_prob
        }
        
        # Create visualizations
        self.visualize_points(points,
                            title='Original Complete Point Cloud',
                            output_file=f"{output_dir}/original_complete.png")
        
        self.visualize_points(final_points,
                            title=f'Processed Complete Point Cloud\n{reduction_percentage:.1f}% reduction',
                            output_file=f"{output_dir}/processed_complete.png")
        
        # Always print key statistics
        print("\nPoint Cloud Processing Results:")
        print(f"Input points:  {stats['original_points']:,}")
        print(f"Output points: {stats['processed_points']:,}")
        print(f"Reduction:     {stats['reduction_percentage']:.1f}%")
        print(f"High anomaly:  {stats['high_anomaly_points']:,}")
        print(f"Low prob/outside std: {stats['low_prob_points']:,}")
        print("\nVoxel Category Breakdown:")
        print(f"Total voxels: {stats['total_voxels']:,}")
        print(f"Too few points: {stats['voxels_too_few_points']:,} ({(stats['voxels_too_few_points']/stats['total_voxels'])*100:.1f}%)")
        print(f"Unimodal:      {stats['voxels_unimodal']:,} ({(stats['voxels_unimodal']/stats['total_voxels'])*100:.1f}%)")
        print(f"Multimodal:    {stats['voxels_multimodal']:,} ({(stats['voxels_multimodal']/stats['total_voxels'])*100:.1f}%)")
        
        # Add point type breakdown
        print("\nPoint Type Breakdown:")
        print(f"Mode/median points: {len(final_points) - stats['high_anomaly_points'] - stats['low_prob_points']:,}")
        print(f"High anomaly points: {stats['high_anomaly_points']:,}")
        print(f"Low prob/outside std: {stats['low_prob_points']:,}")
        
        return final_points, stats

def main():
    """
    Main entry point for the bathymetric point cloud processing script.
    
    This script processes bathymetric point clouds using an isolation forest-based
    approach for denoising and subsampling. It follows IHO S-44 standards for
    bathymetric uncertainty and provides tools for visualization and analysis.
    
    Command line arguments:
        input_file: Path to input XYZ point cloud file
        --output-dir: Directory to save results (default: isolation_forest_results)
        --group-size: Number of points per processing group (default: 1000)
        --voxel-x-size: Size of voxels in X dimension in meters (default: 1.0)
        --voxel-y-size: Size of voxels in Y dimension in meters (default: 1.0)
        --anomaly-threshold: Threshold for anomaly detection (default: 0.5)
        --mode-prob-threshold: Minimum probability for mode assignment (default: 0.3)
        --min-points-mode: Minimum points needed for mode fitting (default: 3)
        --max-modes: Maximum number of modes to fit (default: 1)
        --verbose: Enable verbose output
        --save-intermediate: Save intermediate visualizations
        --plot-interval: Plot every N voxels (default: 500)
        --n-jobs: Number of parallel jobs (-1 for all cores)
        --version: Show version information
    """
    import argparse
    from datetime import datetime
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Process bathymetric point clouds using isolation forest-based denoising and subsampling.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('input_file', 
                       help='Path to input XYZ point cloud file')
    
    # Optional arguments
    parser.add_argument('--output-dir', 
                       default='isolation_forest_results',
                       help='Directory to save results (default: isolation_forest_results)')
    
    # Processing parameters
    parser.add_argument('--group-size', 
                       type=int, 
                       default=1000,
                       help='Number of points per processing group (default: 1000)')
    
    parser.add_argument('--voxel-x-size', 
                       type=float, 
                       default=1.0,
                       help='Size of voxels in X dimension in meters (default: 1.0)')
    
    parser.add_argument('--voxel-y-size', 
                       type=float, 
                       default=1.0,
                       help='Size of voxels in Y dimension in meters (default: 1.0)')
    
    parser.add_argument('--anomaly-threshold', 
                       type=float, 
                       default=0.5,
                       help='Threshold for anomaly detection (default: 0.5)')
    
    parser.add_argument('--mode-prob-threshold', 
                       type=float, 
                       default=0.3,
                       help='Minimum probability for mode assignment (default: 0.3)')
    
    parser.add_argument('--min-points-mode', 
                       type=int, 
                       default=3,
                       help='Minimum points needed for mode fitting (default: 3)')
    
    parser.add_argument('--max-modes', 
                       type=int, 
                       default=1,
                       help='Maximum number of modes to fit (default: 1)')
    
    # Output and logging options
    parser.add_argument('--verbose', 
                       action='store_true',
                       help='Enable verbose output')
    
    parser.add_argument('--save-intermediate', 
                       action='store_true',
                       help='Save intermediate visualizations')
    
    parser.add_argument('--plot-interval', 
                       type=int, 
                       default=500,
                       help='Plot every N voxels (default: 500)')
    
    parser.add_argument('--n-jobs', 
                       type=int, 
                       default=-1,
                       help='Number of parallel jobs (-1 for all cores)')
    
    parser.add_argument('--version', 
                       action='version',
                       version=f'%(prog)s {__version__}',
                       help='Show version information')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save command line arguments
    with open(os.path.join(output_dir, 'parameters.txt'), 'w') as f:
        f.write('Command line arguments:\n')
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')
    
    # Create processor instance
    processor = IsolationGrid(
        group_size=args.group_size,
        voxel_x_size=args.voxel_x_size,
        voxel_y_size=args.voxel_y_size,
        anomaly_threshold=args.anomaly_threshold,
        mode_probability_threshold=args.mode_prob_threshold,
        min_points_for_mode=args.min_points_mode,
        max_modes=args.max_modes,
        verbose=args.verbose,
        save_intermediate_files=args.save_intermediate,
        plot_interval=args.plot_interval
    )
    
    # Process point cloud
    try:
        print(f"\nProcessing point cloud: {args.input_file}")
        print(f"Output directory: {output_dir}")
        print(f"Number of parallel jobs: {args.n_jobs}")
        print("\nParameters:")
        print(f"Group size: {args.group_size:,} points")
        print(f"Voxel size: {args.voxel_x_size:.2f}m x {args.voxel_y_size:.2f}m")
        print(f"Anomaly threshold: {args.anomaly_threshold:.2f}")
        print(f"Mode probability threshold: {args.mode_prob_threshold:.2f}")
        print(f"Minimum points for mode: {args.min_points_mode}")
        print(f"Maximum modes: {args.max_modes}")
        
        final_points, stats = processor.process(args.input_file, output_dir, args.n_jobs)
        
        # Save processing statistics
        with open(os.path.join(output_dir, 'statistics.txt'), 'w') as f:
            f.write('Processing Statistics:\n')
            for key, value in stats.items():
                f.write(f'{key}: {value}\n')
        
        print("\nProcessing completed successfully!")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"\nError processing point cloud: {str(e)}")
        raise

if __name__ == "__main__":
    main()


