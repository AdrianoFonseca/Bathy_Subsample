"""Command-line interface for BathySubsample."""

# Update imports
from bathy_subsample.version import __version__
from bathy_subsample.core.bathy_subsample import BathySubsample

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
    processor = BathySubsample(
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