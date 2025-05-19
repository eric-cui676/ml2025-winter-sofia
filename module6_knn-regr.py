from __future__ import annotations
import sys
from typing import Tuple

import numpy as np


def read_positive_int(prompt: str) -> int:
    """Read a positive integer from stdin, re‑prompt until valid."""
    while True:
        try:
            value = int(input(prompt))
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            print("Please enter a positive integer.")


def read_float(prompt: str) -> float:
    """Read a float from stdin, re‑prompt until valid."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a valid number (real).")


def knn_regression(points: np.ndarray, k: int, query_x: float) -> float:
    """Return k‑NN regression prediction for query_x.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) where column 0 is x and column 1 is y.
    k : int
        Number of neighbours to use (must satisfy 1 ≤ k ≤ N).
    query_x : float
        The x‑coordinate at which to predict y.

    Returns
    -------
    float
        Predicted y value (mean of y of k nearest neighbours).
    """
    # 1‑D Euclidean distance simplifies to absolute difference.
    distances = np.abs(points[:, 0] - query_x)
    nearest_idx = np.argsort(distances)[:k]
    return points[nearest_idx, 1].mean()


def main() -> None:
    N = read_positive_int("Enter number of points N: ")
    k = read_positive_int("Enter number of neighbours k: ")

    if k > N:
        print("Error: k cannot be greater than N.")
        sys.exit(1)

    # Pre‑allocate an (N, 2) float array for efficiency
    points = np.empty((N, 2), dtype=float)

    for i in range(N):
        points[i, 0] = read_float(f"Enter x{i + 1}: ")
        points[i, 1] = read_float(f"Enter y{i + 1}: ")

    query_x = read_float("Enter query X: ")

    prediction = knn_regression(points, k, query_x)
    print(f"Predicted Y: {prediction}")


if __name__ == "__main__":
    main()
