"""Main IsolationGrid class implementation."""

import os
import numpy as np
from sklearn.neighbors import KDTree
from collections import defaultdict
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import warnings
from sklearn.exceptions import ConvergenceWarning

# Local imports
import eif as iso
from bathy_subsample.utils.visualization import visualize_points, visualize_leaf
from bathy_subsample.utils.geometry import calculate_tvu
from bathy_subsample.core.normal_estimation import estimate_normals
from bathy_subsample.core.voxel import create_voxel_grid, extract_all_modes

class BathySubsample:
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
        Initialize the BathySubsample processor.
        
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
    
    def calculate_tvu(self, depth):
        """Calculate Total Vertical Uncertainty for a given depth."""
        return calculate_tvu(depth)
    
    def estimate_normals(self, points, k=20):
        """Estimate normal vectors using points within ±3*TVU of median depth."""
        return estimate_normals(points, k, self.calculate_tvu, self.verbose)
    
    def create_voxel_grid(self, points):
        """Create a voxel grid aligned with surface normal."""
        return create_voxel_grid(points, self.voxel_x_size, self.voxel_y_size, 
                                self.estimate_normals, self.calculate_tvu, self.verbose)
    
    def extract_all_modes(self, data):
        """Extract modes from depth values using Gaussian Mixture Model."""
        return extract_all_modes(data, self.min_points_for_mode, 
                               self.max_modes, self.calculate_tvu)

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
            visualize_leaf(points, leaf_id, output_dir, selected_points, high_anomaly_points, mode_points, low_prob_points)
        
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
        visualize_points(points,
                        title='Original Complete Point Cloud',
                        output_file=f"{output_dir}/original_complete.png")
        
        visualize_points(final_points,
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