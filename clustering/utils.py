"""
This is utils.py file and contain all the utility functions and classes that need in both of our algos
"""

import math
import random
from PIL import Image
import numpy as np
import os


# ___________ GENERIC DISTANCE FUNCTIONS ____________
def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.

    Args:
        point1, point2: Lists of numbers (same length)

    Returns:
        float: Euclidean distance

    """
    # Check if points have same dimensions
    if len(point1) != len(point2):
        raise ValueError(f"Points must have same dimensions. Got {len(point1)} and {len(point2)}")

    # Sum of squared differences
    sum_squared = 0
    for i in range(len(point1)):
        diff = point1[i] - point2[i]
        sum_squared += diff * diff  # diff²

    return math.sqrt(sum_squared)


def manhattan_distance(point1, point2):
    """
    Calculate Manhattan distance between two points.
    """
    if len(point1) != len(point2):
        raise ValueError(f"Points must have same dimensions. Got {len(point1)} and {len(point2)}")

    sum_absolute = 0
    for i in range(len(point1)):
        sum_absolute += abs(point1[i] - point2[i])

    return sum_absolute


def maximum_distance(point1, point2):
    """
    Calculate Maximum distance between two points.
    """
    if len(point1) != len(point2):
        raise ValueError(f"Points must have same dimensions. Got {len(point1)} and {len(point2)}")

    max_diff = 0
    for i in range(len(point1)):
        diff = abs(point1[i] - point2[i])
        if diff > max_diff:
            max_diff = diff

    return max_diff


def get_distance_function(name):
    """
    Get distance function by name.

    Args:
        name: 'euclidean', 'manhattan', or 'maximum'

    Returns:
        function: The distance function

    Raises:
        ValueError: If name is not recognized
    """
    distances = {
        'euclidean': euclidean_distance,
        'manhattan': manhattan_distance,
        'maximum': maximum_distance,
        'euclid': euclidean_distance,  # alias
        'man': manhattan_distance,  # alias
        'max': maximum_distance,  # alias
        'chebyshev': maximum_distance  # alias
    }

    name_lower = name.lower()
    if name_lower not in distances:
        raise ValueError(f"Unknown distance function: {name}. "
                         f"Available: {list(distances.keys())}")

    return distances[name_lower]


# CLUSTER CLASS
class Cluster:
    """Represents a cluster of points with a centroid."""

    def __init__(self, centroid):
        """
        Initialize a cluster with a centroid.

        Args:
            centroid: Center point of the cluster
        """
        self.centroid = centroid.copy() if hasattr(centroid, 'copy') else list(centroid)
        self.points = []  # Points belonging to this cluster
        self.point_indices = []  # Indices of points (for efficiency)

    def __repr__(self):
        """String representation for debugging."""
        return f"Cluster(centroid={[round(c, 2) for c in self.centroid]}, points={len(self.points)})"

    def clear_points(self):
        """Clear points for next iteration."""
        self.points = []
        self.point_indices = []

    def add_point(self, point, index):
        """
        Add a point to the cluster.

        Args:
            point: The data point
            index: Position in original dataset
        """
        self.points.append(point)
        self.point_indices.append(index)

    def update_centroid(self):
        """
        Calculate new centroid as average of all points.

        Returns:
            List[float]: The new centroid
        """
        if not self.points:
            return self.centroid  # No points, keep old centroid

        # Initialize sum with zeros
        new_centroid = [0.0] * len(self.centroid)

        # Sum all points
        for point in self.points:
            for i in range(len(point)):
                new_centroid[i] += point[i]

        # Divide by number of points to get average
        for i in range(len(new_centroid)):
            new_centroid[i] /= len(self.points)

        self.centroid = new_centroid
        return new_centroid

    def calculate_sse(self, distance_func):
        """
        Calculate Sum of Squared Errors for this cluster.

        Args:
            distance_func: Function to calculate distance

        Returns:
            float: SSE value
        """
        sse = 0.0
        for point in self.points:
            dist = distance_func(point, self.centroid)
            sse += dist * dist
        return sse


def calculate_centroid(points):
    """
    Calculate centroid (mean) of a list of points.

    Args:
        points: List of points (each point is a list of numbers)

    Returns:
        List[float]: Centroid point
    """
    if not points:
        raise ValueError("Cannot calculate centroid of empty list")

    # Determine dimensionality from first point
    num_dimensions = len(points[0])

    # Initialize sum array
    centroid = [0.0] * num_dimensions

    # Sum all points
    for point in points:
        if len(point) != num_dimensions:
            raise ValueError(f"All points must have same dimension. "
                             f"Expected {num_dimensions}, got {len(point)}")
        for i in range(num_dimensions):
            centroid[i] += point[i]

    # Divide by number of points
    num_points = len(points)
    for i in range(num_dimensions):
        centroid[i] /= num_points

    return centroid


def initialize_random_centroids(points, k, seed=None):
    """
    Randomly initialize k centroids from points.

    Args:
        points: Data points
        k: Number of centroids
        seed: Random seed for reproducibility

    Returns:
        List[List[float]]: k random centroids
    """
    if seed is not None:
        random.seed(seed)

    if k <= 0:
        raise ValueError(f"k must be positive. Got {k}")

    if k > len(points):
        raise ValueError(f"Cannot select {k} centroids from {len(points)} points")

    # Randomly choose k distinct points as initial centroids
    indices = random.sample(range(len(points)), k)
    centroids = [points[i].copy() if hasattr(points[i], 'copy') else list(points[i])
                 for i in indices]

    return centroids


def find_nearest_centroid(point, centroids, distance_func):
    """
    Find the nearest centroid to a point.

    Args:
        point: The point
        centroids: List of centroids
        distance_func: Distance function

    Returns:
        Tuple[int, float]: (index of nearest centroid, distance)
    """
    min_distance = float('inf')
    nearest_idx = -1

    for i, centroid in enumerate(centroids):
        dist = distance_func(point, centroid)
        if dist < min_distance:
            min_distance = dist
            nearest_idx = i

    return nearest_idx, min_distance

# This function is optional
def normalize_points(points):
    """
    Normalize points to [0, 1] range per dimension.

    Args:
        points: List of points

    Returns:
        List[List[float]]: Normalized points
    """
    if not points:
        return []

    num_dimensions = len(points[0])
    num_points = len(points)

    # Find min and max for each dimension
    mins = [float('inf')] * num_dimensions
    maxs = [float('-inf')] * num_dimensions

    for point in points:
        for i in range(num_dimensions):
            if point[i] < mins[i]:
                mins[i] = point[i]
            if point[i] > maxs[i]:
                maxs[i] = point[i]

    # Normalize each point
    normalized = []
    ranges = [maxs[i] - mins[i] for i in range(num_dimensions)]

    for point in points:
        norm_point = []
        for i in range(num_dimensions):
            if ranges[i] == 0:
                norm_point.append(0.0)  # All values are same
            else:
                norm_point.append((point[i] - mins[i]) / ranges[i])
        normalized.append(norm_point)

    return normalized


# IMAGE Utility FUNCTIONS
def load_image(image_path):
    """
    Load an image and convert to list of RGB pixels (Data points).

    Args:
        image_path: Path to image file

    Returns:
        tuple: (pixels_list, width, height, original_image)

    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image cannot be loaded
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        # Open image
        img = Image.open(image_path)

        # Convert to RGB if not already
        if img.mode != 'RGB':
            print(f"  Converting image from {img.mode} to RGB")
            img = img.convert('RGB')

        # Get dimensions
        width, height = img.size

        # Convert to NumPy array for efficiency
        img_array = np.array(img, dtype=np.float32)
        pixels = img_array.reshape(-1, 3).tolist()

        return pixels, width, height, img

    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {str(e)}")


def save_segmented_image(labels, centroids, width, height, output_path):
    """
    Create output image where each pixel is colored by its cluster's average color.

    Args:
        labels: List of cluster labels for each pixel
        centroids: List of cluster centroids (average colors)
        width, height: Original image dimensions
        output_path: Where to save the result
    """
    # Convert centroids to integers (0-255 range)
    int_centroids = []
    for centroid in centroids:
        int_centroid = [int(round(c)) for c in centroid]
        # Clamp to 0-255 range
        int_centroid = [max(0, min(255, c)) for c in int_centroid]
        int_centroids.append(int_centroid)

    # Create output array
    output_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Fill with cluster colors
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            cluster_idx = labels[idx]
            output_array[y, x] = int_centroids[cluster_idx]

    # Save image
    output_img = Image.fromarray(output_array)
    output_img.save(output_path)
    print(f"  Saved clustered image to: {output_path}")


def show_image_info(image_path):
    """
    Display detailed information about an image.

    Args:
        image_path: Path to image

    Returns:
        dict: Image information
    """
    try:
        img = Image.open(image_path)
        info = {
            'path': image_path,
            'size': img.size,
            'mode': img.mode,
            'format': img.format,
            'total_pixels': img.size[0] * img.size[1]
        }

        print(f"\nImage Information:")
        print(f"  Path: {info['path']}")
        print(f"  Size: {info['size'][0]}x{info['size'][1]}")
        print(f"  Mode: {info['mode']}")
        print(f"  Format: {info['format']}")
        print(f"  Total Pixels: {info['total_pixels']:,}")

        # Show color statistics if RGB
        if img.mode == 'RGB':
            img_array = np.array(img)
            print(f"\n  Color Statistics:")
            print(f"    Red:   {img_array[:, :, 0].min():3d} - {img_array[:, :, 0].max():3d}")
            print(f"    Green: {img_array[:, :, 1].min():3d} - {img_array[:, :, 1].max():3d}")
            print(f"    Blue:  {img_array[:, :, 2].min():3d} - {img_array[:, :, 2].max():3d}")

        return info

    except Exception as e:
        print(f"Error loading image: {e}")
        return None

