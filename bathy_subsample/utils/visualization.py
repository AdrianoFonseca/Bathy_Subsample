"""Visualization functions for point clouds and leaf data."""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def visualize_points(points, scores=None, title="Point Cloud", 
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

def visualize_leaf(points, leaf_id, output_dir, selected_points, high_anomaly_points, mode_points, low_prob_points):
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