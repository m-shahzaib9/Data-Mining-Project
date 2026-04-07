"""
main.py - Main command-line interface for clustering algorithms
"""

import argparse
import sys
import os
import time

# Import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_image, save_segmented_image, show_image_info
from k_mean_algo import KMeans
from dbscan_algo import DBSCAN


def parse_arguments():
    """Parse command-line arguments provided by user."""
    parser = argparse.ArgumentParser(
        description='Image Segmentation using Clustering Algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
            Examples:
              # K-Means with 5 clusters, Euclidean distance by default
              python main.py --input 1_kmeans.png --output result.png --algorithm kmeans --k 5
            
              # K-Means with Manhattan distance, 10 iterations max
              python main.py --input 1_kmeans.png --output result.png --algorithm kmeans --k 8 --distance manhattan --max-iter 10
            
              # To show the image information
              python main.py --input image.jpg --info
        '''
    )

    # Required arguments
    parser.add_argument('--input', required=True, help='Input the path of the image')
    parser.add_argument('--output', help='Output image path (for clustering)')

    # Algorithm selection
    parser.add_argument('--algorithm', choices=['kmeans', 'dbscan'],
                        default='kmeans', help='Which clustering algorithm to use?')

    # K-Means specific
    parser.add_argument('--k', type=int, default=5,
                        help='Number of clusters for K-Means (default: 5)')
    parser.add_argument('--distance', choices=['euclidean', 'manhattan', 'maximum'],
                        default='euclidean', help='Distance metric (default: euclidean)')
    parser.add_argument('--max-iter', type=int, default=100,
                        help='Maximum iterations for K-Means (default: 100)')
    parser.add_argument('--tol', type=float, default=1e-4,
                        help='Convergence tolerance (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    # DBSCAN specific
    parser.add_argument('--eps', type=float, default=30.0,
                        help='Maximum distance between neighbors for DBSCAN (default: 30.0)')
    parser.add_argument('--min-samples', type=int, default=10,
                        help='Minimum samples to form a cluster for DBSCAN (default: 10)')

    # Other options
    parser.add_argument('--info', action='store_true',
                        help='Show image information and exit')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed progress information')

    return parser.parse_args()


def run_kmeans_on_image(input_path: str, output_path: str, k: int = 5,
                        distance: str = 'euclidean', max_iter: int = 100,
                        tol: float = 1e-4, seed: int = None,
                        verbose: bool = False) -> dict:
    """
    Run K-Means clustering on an image.

    Args:
        input_path: Input image path
        output_path: Output image path
        k: Number of clusters for kmeans
        distance: Distance function
        max_iter: Maximum iterations
        tol: Convergence tolerance
        seed: Random seed
        verbose: Show detailed output

    Returns:
        dict: Clustering results and metrics
    """
    if verbose:
        print(f"Loading image: {input_path}")

    # Load image
    pixels, width, height, img = load_image(input_path)

    if verbose:
        print(f"  Image size: {width}x{height} ({len(pixels)} pixels)")
        print(f"  Running K-Means with k={k}, distance={distance}")
        print("  Clustering in progress...")

    start_time = time.time()

    # Run K-Means clustering
    kmeans = KMeans(k=k, distance=distance, max_iter=max_iter,
                    tol=tol, random_state=seed)
    kmeans.fitting(pixels)

    clustering_time = time.time() - start_time

    if verbose:
        print(f"  Clustering completed in {clustering_time:.2f} seconds")
        print(f"  Converged in {kmeans.n_iter} iterations")
        print(f"  Final inertia (SSE): {kmeans.inertia:,.2f}")

    # Save clustered image
    if verbose:
        print(f"  Saving result to: {output_path}")

    save_segmented_image(kmeans.labels, kmeans.centroids, width, height, output_path)

    # Return results
    results = {
        'algorithm': 'kmeans',
        'k': k,
        'distance': distance,
        'n_iterations': kmeans.n_iter,
        'inertia': kmeans.inertia,
        'clustering_time': clustering_time,
        'input_image': input_path,
        'output_image': output_path,
        'image_size': (width, height),
        'n_pixels': len(pixels),
        'cluster_sizes': [len(c.points) for c in kmeans.clusters],
        'centroids': kmeans.centroids
    }

    return results


def run_dbscan_on_image(input_path: str, output_path: str, eps: float = 30.0,
                        min_samples: int = 10, distance: str = 'euclidean',
                        verbose: bool = False) -> dict:
    """
    Run DBSCAN clustering on an image.

    Args:
        input_path: Input image path
        output_path: Output image path
        eps: Maximum distance between neighbors
        min_samples: Minimum points to form a cluster
        distance: Distance function
        verbose: Show detailed output

    Returns:
        dict: Clustering results and metrics
    """
    if verbose:
        print(f"Loading image: {input_path}")

    # Load image
    pixels, width, height, img = load_image(input_path)

    if verbose:
        print(f"  Image size: {width}x{height} ({len(pixels)} pixels)")
        print(f"  Running DBSCAN with eps={eps}, min_samples={min_samples}")
        print("  Clustering in progress...")

    start_time = time.time()

    # Run DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, distance=distance)
    dbscan.fitting(pixels)

    clustering_time = time.time() - start_time

    if verbose:
        print(f"  Clustering completed in {clustering_time:.2f} seconds")
        print(f"  Found {dbscan.n_clusters} clusters")
        print(f"  Core points: {len(dbscan.core_points)}")

    # For DBSCAN, we need to handle noise points
    # We'll color noise points as black (0,0,0)
    cluster_info = dbscan.get_cluster_info(pixels)

    # Prepare centroids for saving (noise will be black)
    centroids = []
    for i in range(1, dbscan.n_clusters + 1):
        if i in cluster_info['centroids']:
            centroids.append(cluster_info['centroids'][i])
        else:
            centroids.append([0, 0, 0])  # Fallback

    # Add black color for noise at the end
    centroids.append([0, 0, 0])  # Index for noise

    # Adjust labels: noise becomes last "cluster" index
    labels = []
    for label in dbscan.labels:
        if label == -1:  # Noise
            labels.append(len(centroids) - 1)
        else:
            labels.append(label - 1)

    if verbose:
        print(f"  Saving result to: {output_path}")

    save_segmented_image(labels, centroids, width, height, output_path)

    # Return results
    results = {
        'algorithm': 'dbscan',
        'eps': eps,
        'min_samples': min_samples,
        'distance': distance,
        'n_clusters': dbscan.n_clusters,
        'n_core_points': len(dbscan.core_points),
        'n_noise_points': cluster_info['n_noise_points'],
        'clustering_time': clustering_time,
        'input_image': input_path,
        'output_image': output_path,
        'image_size': (width, height),
        'n_pixels': len(pixels),
        'cluster_sizes': cluster_info['cluster_sizes'],
        'centroids': centroids[:-1]  # Exclude noise
    }

    return results


def display_results(results: dict):
    """Display clustering results in a readable format."""
    print("\n" + "=" * 60)
    print("CLUSTERING RESULTS")
    print("=" * 60)

    print(f"Algorithm: {results['algorithm'].upper()}")
    print(f"Input: {results['input_image']}")
    print(f"Output: {results['output_image']}")
    print(f"Image size: {results['image_size'][0]}x{results['image_size'][1]}")
    print(f"Number of pixels: {results['n_pixels']:,}")

    if results['algorithm'] == 'kmeans':
        print(f"\nK-Means Parameters:")
        print(f"  k (clusters): {results['k']}")
        print(f"  Distance function: {results['distance']}")
        print(f"  Iterations: {results['n_iterations']}")
        print(f"  Inertia (SSE): {results['inertia']:,.2f}")
        print(f"  Clustering time: {results['clustering_time']:.2f} seconds")

        print(f"\nCluster Information:")
        for i, (size, centroid) in enumerate(zip(results['cluster_sizes'], results['centroids'])):
            # Convert centroid to integers for display (RGB colors)
            int_centroid = [int(round(c)) for c in centroid]
            print(f"  Cluster {i}: {size:6,d} pixels, color: RGB{tuple(int_centroid)}")
    elif results['algorithm'] == 'dbscan':
        print(f"\nDBSCAN Parameters:")
        print(f"  eps (neighborhood radius): {results['eps']}")
        print(f"  min_samples (density threshold): {results['min_samples']}")
        print(f"  Distance function: {results['distance']}")
        print(f"  Clustering time: {results['clustering_time']:.2f} seconds")

        print(f"\nDBSCAN Results:")
        print(f"  Number of clusters found: {results['n_clusters']}")
        print(f"  Core points: {results['n_core_points']:,}")
        print(f"  Noise points: {results['n_noise_points']:,} (colored black)")

        print(f"\nCluster Information:")
        for cluster_id, size in results['cluster_sizes'].items():
            if cluster_id in results.get('centroid_indices', {}):
                centroid_idx = results['centroid_indices'][cluster_id]
                if centroid_idx < len(results['centroids']):
                    centroid = results['centroids'][centroid_idx]
                    int_centroid = [int(round(c)) for c in centroid]
                    print(f"  Cluster {cluster_id}: {size:6,d} pixels, color: RGB{tuple(int_centroid)}")
                else:
                    print(f"  Cluster {cluster_id}: {size:6,d} pixels")
            else:
                print(f"  Cluster {cluster_id}: {size:6,d} pixels")

    print("=" * 60)


def main():
    """Starting point of the program. Main function!"""
    args = parse_arguments()

    # Show image information if user request
    if args.info:
        show_image_info(args.input)
        return

    # Check if output path is provided for clustering (required)
    if not args.output:
        print("Error: Output path is required for clustering")
        print("Use --output <path> to specify where to save the result")
        return

    # Run clustering based on selected algorithm
    if args.algorithm == 'kmeans':
        if args.verbose:
            print("Starting K-Means clustering...")
            print(f"Input: {args.input}")
            print(f"Output: {args.output}")
            print(f"Parameters: k={args.k}, distance={args.distance}")

        try:
            results = run_kmeans_on_image(
                input_path=args.input,
                output_path=args.output,
                k=args.k,
                distance=args.distance,
                max_iter=args.max_iter,
                tol=args.tol,
                seed=args.seed,
                verbose=args.verbose
            )

            display_results(results)

        except Exception as e:
            print(f"Error during K-Means clustering: {e}")
            sys.exit(1)

    elif args.algorithm == 'dbscan':
        if args.verbose:
            print("Starting DBSCAN clustering...")
            print(f"Input: {args.input}")
            print(f"Output: {args.output}")
            print(f"Parameters: eps={args.eps}, min_samples={args.min_samples}")

        try:
            results = run_dbscan_on_image(
                input_path=args.input,
                output_path=args.output,
                eps=args.eps,
                min_samples=args.min_samples,
                distance=args.distance,
                verbose=args.verbose
            )

            display_results(results)

        except Exception as e:
            print(f"Error during DBSCAN clustering: {e}")
            sys.exit(1)

    else:
        print(f"Unknown algorithm please provide the correct one: {args.algorithm}")
        sys.exit(1)


if __name__ == "__main__":
    main()