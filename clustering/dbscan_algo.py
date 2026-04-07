"""
dbscan_algo.py
This file contain the implementation of the DBSCAN algorithm.
"""

import sys
import os
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_distance_function


class DBSCAN:
    """
    DBSCAN clustering algorithm.

    Attributes:
        core_points (set): Indices of core points
        eps (float): Maximum distance between two points to be considered neighbors
        min_samples (int): Minimum number of points to form a dense region
        distance_func (callable): Distance function
        labels (list): Cluster labels for each point (-1 for noise)

    """

    # Constants for point states
    UNVISITED = 0
    VISITED = 1
    NOISE = -1

    def __init__(self, eps: float = 0.5, min_samples: int = 5,
                 distance: str = 'euclidean'):
        """
        Initialize DBSCAN algorithm.

        Args:
            eps: Maximum distance between two points to be considered neighbors
            min_samples: Minimum number of points to form a dense region
            distance: Distance function name ('euclidean', 'manhattan', 'maximum')
        """
        if eps <= 0:
            raise ValueError(f"eps must be positive. Got {eps}")
        if min_samples <= 0:
            raise ValueError(f"min_samples must be positive. Got {min_samples}")

        self.eps = eps
        self.min_samples = min_samples
        self.distance_func = get_distance_function(distance)

        # Results
        self.labels = []
        self.core_points = set()
        self.n_clusters = 0

    def fitting(self, X: List[List[float]]) -> 'DBSCAN':
        """
        Fit the DBSCAN to the data.

        Args:
            X: List of data points (each point is a list of numbers)

        Returns:
            self: Fitted object of DBSCAN
        """
        if not X:
            raise ValueError("Cannot fit on empty data")

        n_points = len(X)

        # Initialize all points as unvisited (-1 means noise/unassigned)
        self.labels = [-1] * n_points
        self.core_points = set()
        self.n_clusters = 0

        # Main DBSCAN algorithm
        for i in range(n_points):
            # Skip if already visited (assigned to cluster)
            if self.labels[i] != -1:
                continue

            # Find all neighbors within eps distance
            neighbors = self._range_query(X, i)

            # Check if point is a core point
            if len(neighbors) < self.min_samples:
                # Point is noise (for now, may become border point later)
                self.labels[i] = self.NOISE
            else:
                # Point is a core point - start a new cluster
                self.n_clusters += 1
                self.core_points.add(i)
                self.labels[i] = self.n_clusters

                # Expand the cluster
                self._expand_cluster(X, i, neighbors)

        return self

    def _range_query(self, X: List[List[float]], point_idx: int) -> List[int]:
        """
        Find all points within eps distance of the given point.

        Args:
            X: All data points
            point_idx: Index of the query point

        Returns:
            List[int]: Indices of neighbors within eps distance
        """
        neighbors = []
        query_point = X[point_idx]

        for j, other_point in enumerate(X):
            if point_idx == j:
                continue

            distance = self.distance_func(query_point, other_point)
            if distance <= self.eps:
                neighbors.append(j)

        return neighbors

    def _expand_cluster(self, X: List[List[float]], point_idx: int,
                        neighbors: List[int]) -> None:
        """
        Expand a cluster from a core point.

        Args:
            X: All data points
            point_idx: Index of core point
            neighbors: Initial neighbors
        """
        cluster_id = self.labels[point_idx]
        queue = neighbors.copy()

        while queue:
            neighbor_idx = queue.pop(0)

            # If point was noise, make it border point
            if self.labels[neighbor_idx] == -1:
                self.labels[neighbor_idx] = cluster_id
            # Skip if already assigned to this or another cluster
            elif self.labels[neighbor_idx] != -1:
                continue
            else:
                # Assign to current cluster
                self.labels[neighbor_idx] = cluster_id

            # Find this point's neighbors
            neighbor_neighbors = self._range_query(X, neighbor_idx)

            # Check if it's also a core point
            if len(neighbor_neighbors) >= self.min_samples:
                self.core_points.add(neighbor_idx)

                # Add new neighbors to queue
                for new_neighbor in neighbor_neighbors:
                    if self.labels[new_neighbor] == -1 and new_neighbor not in queue:
                        queue.append(new_neighbor)

    def get_cluster_info(self, X: List[List[float]]) -> dict:
        """
        Get information about each cluster.

        Args:
            X: Data points (needed to calculate statistics)

        Returns:
            dict: Cluster information
        """
        if not self.labels:
            raise ValueError("Model not fitted yet. Call fitting() first.")

        # Count points per cluster
        cluster_counts = {}
        noise_count = 0

        for label in self.labels:
            if label == self.NOISE:
                noise_count += 1
            else:
                cluster_counts[label] = cluster_counts.get(label, 0) + 1

        # Calculate centroids for each cluster
        cluster_points = {}
        for i, label in enumerate(self.labels):
            if label != self.NOISE:
                if label not in cluster_points:
                    cluster_points[label] = []
                cluster_points[label].append(X[i])

        centroids = {}
        for label, points in cluster_points.items():
            # Calculate centroid (mean) of points in cluster
            centroid = [0.0] * len(points[0])
            for point in points:
                for dim in range(len(point)):
                    centroid[dim] += point[dim]

            for dim in range(len(centroid)):
                centroid[dim] /= len(points)

            centroids[label] = centroid

        info = {
            'algorithm': 'dbscan',
            'eps': self.eps,
            'min_samples': self.min_samples,
            'n_clusters': self.n_clusters,
            'n_core_points': len(self.core_points),
            'n_noise_points': noise_count,
            'cluster_sizes': cluster_counts,
            'centroids': centroids
        }

        return info


# Helper function for easy usage
def dbscan_cluster(points: List[List[float]],
                   eps: float = 0.5,
                   min_samples: int = 5,
                   distance: str = 'euclidean') -> Tuple[List[int], dict]:
    """
    Convenience function for DBSCAN clustering.

    Args:
        points: Data points to cluster
        eps: Maximum distance between neighbors
        min_samples: Minimum points to form a cluster
        distance: Distance function name

    Returns:
        Tuple[List[int], dict]: (labels, cluster_info)
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, distance=distance)
    dbscan.fitting(points)
    cluster_info = dbscan.get_cluster_info(points)
    return dbscan.labels, cluster_info

