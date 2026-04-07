"""
k_mean_algo.py
This file contain the implementation of the k-means algorithm.
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Optional

# Import Phase 1 components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    Cluster,
    initialize_random_centroids,
    find_nearest_centroid,
    get_distance_function,
)


class KMeans:
    """
    K-Means clustering algorithm.

    Attributes:
        k (int): Number of clusters
        distance_func (callable): Distance function
        max_iter (int): Maximum iterations
        random_state (int): Random seed
        clusters (list): List of Cluster objects
        labels (list): Cluster labels for each point
        inertia (float): Sum of squared distances
        centroids (list): Current centroids
        tol (float): Convergence tolerance

    """

    def __init__(self, k: int = 3,
                 distance: str = 'euclidean',
                 max_iter: int = 100,
                 tol: float = 1e-4,
                 random_state: Optional[int] = None):
        """
        Initialize K-Means algorithm.

        Args:
            k: Number of clusters
            distance: Distance function name ('euclidean', 'manhattan', 'maximum')
            max_iter: Maximum number of iterations
            tol: Convergence tolerance (centroid change threshold)
            random_state: Random seed for reproducibility
        """
        if k <= 0:
            raise ValueError(f"k must be positive. Got {k}")

        self.k = k
        self.distance_func = get_distance_function(distance)
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Results
        self.clusters = []
        self.labels = []
        self.centroids = []
        self.inertia = 0.0
        self.n_iter = 0

    def fitting(self, X: List[List[float]]) -> 'KMeans':
        """
        Fit K-Means algo to the data.

        Args:
            X: List of data points (each point is a list of numbers)

        Returns:
            self: Fitted KMeans object
        """
        if not X:
            raise ValueError("Cannot fit on empty data")

        n_points = len(X)
        if self.k > n_points:
            raise ValueError(f"Cannot cluster {n_points} points into {self.k} clusters")

        # Step 1: Initialize centroids randomly
        self.centroids = initialize_random_centroids(X, self.k, self.random_state)

        # Initialize clusters
        self.clusters = [Cluster(centroid) for centroid in self.centroids]

        # Main K-Means loop
        for iteration in range(self.max_iter):
            self.n_iter = iteration + 1

            # Step 2: Assignment - Clear old points and assign new ones
            self._assign_points(X)

            # Step 3: Update - Calculate new centroids
            old_centroids = [c.centroid.copy() for c in self.clusters]
            self._update_centroids()

            # Step 4: Check convergence
            if self._has_converged(old_centroids):
                print(f"Converged after {iteration + 1} iterations")
                break

        # Calculate final inertia (SSE)
        self._calculate_inertia(X)

        return self

    def _assign_points(self, X: List[List[float]]) -> None:
        """
        Assign each point to the nearest cluster.

        Args:
            X: Data points
        """
        # Clear old points from clusters
        for cluster in self.clusters:
            cluster.clear_points()

        # Reset labels
        self.labels = []

        # Assign each point to nearest cluster
        for i, point in enumerate(X):
            nearest_idx, _ = find_nearest_centroid(point, self.centroids, self.distance_func)
            self.labels.append(nearest_idx)
            self.clusters[nearest_idx].add_point(point, i)

    def _update_centroids(self) -> None:
        """
        Update centroids based on assigned points .
        """
        new_centroids = []

        for cluster in self.clusters:
            if cluster.points:
                new_centroid = cluster.update_centroid()
                new_centroids.append(new_centroid)
            else:
                # Empty cluster - keep old centroid
                new_centroids.append(cluster.centroid)

        self.centroids = new_centroids

    def _has_converged(self, old_centroids: List[List[float]]) -> bool:
        """
        Check if algorithm has converged.

        Args:
            old_centroids: Centroids from previous iteration

        Returns:
            bool: True if converged, False otherwise
        """
        if not old_centroids:
            return False

        max_change = 0.0

        for old, new in zip(old_centroids, self.centroids):
            # Calculate Euclidean distance between old and new centroid
            change = 0.0
            for i in range(len(old)):
                diff = old[i] - new[i]
                change += diff * diff
            change = np.sqrt(change)

            if change > max_change:
                max_change = change

        # Converged if maximum centroid change is below tolerance
        return max_change < self.tol

    def _calculate_inertia(self, X: List[List[float]]) -> None:
        """
        Calculate inertia (sum of squared distances to nearest centroid).

        Args:
            X: Data points
        """
        self.inertia = 0.0

        for i, point in enumerate(X):
            cluster_idx = self.labels[i]
            centroid = self.centroids[cluster_idx]
            dist = self.distance_func(point, centroid)
            self.inertia += dist * dist


    def get_cluster_info(self) -> dict:
        """
        Get information about each cluster.

        Returns:
            dict: Cluster information
        """
        info = {
            'n_clusters': self.k,
            'n_iterations': self.n_iter,
            'inertia': self.inertia,
            'clusters': []
        }

        for i, cluster in enumerate(self.clusters):
            cluster_info = {
                'id': i,
                'size': len(cluster.points),
                'centroid': cluster.centroid,
                'sse': cluster.calculate_sse(self.distance_func)
            }
            info['clusters'].append(cluster_info)

        return info


# Helper function for easy usage
def kmeans_cluster(points: List[List[float]],
                   k: int = 3,
                   distance: str = 'euclidean',
                   max_iter: int = 100,
                   tol: float = 1e-4,
                   random_state: Optional[int] = None) -> Tuple[List[int], List[List[float]]]:
    """
    Convenience function for K-Means clustering.

    Args:
        points: Data points to cluster
        k: Number of clusters
        distance: Distance function name
        max_iter: Maximum iterations
        tol: Convergence tolerance
        random_state: Random seed

    Returns:
        Tuple[List[int], List[List[float]]]: (labels, centroids)
    """
    kmeans = KMeans(k=k, distance=distance, max_iter=max_iter,
                    tol=tol, random_state=random_state)
    kmeans.fitting(points)
    return kmeans.labels, kmeans.centroids
