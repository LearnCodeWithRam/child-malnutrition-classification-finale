"""
tps_math.py - Mathematical Foundations of Thin Plate Spline (TPS) Transformations
===================================================================================
Authentic implementation of 2D and 3D Thin Plate Spline (TPS) deformation based on:
1. Bookstein, F. L. (1989). "Principal Warps: Thin-plate splines and the decomposition
   of deformations". IEEE Transactions on Pattern Analysis and Machine Intelligence.
2. Tang, Z., Chen, K., Pan, M., Wang, M., & Song, Z. (2019). "An Augmentation Strategy 
   for Medical Image Processing Based on Statistical Shape Model and 3D Thin Plate Spline 
   for Deep Learning". IEEE Access, 7, 133111-133121.
3. Rème, R. (raphaelreme/tps): Generalized Polyharmonic and Thin Plate Spline Interpolation.
"""

import numpy as np
from typing import Tuple, Optional, Union


def tps_kernel_2d(r: np.ndarray) -> np.ndarray:
    """
    Radial basis function kernel for 2D Thin Plate Spline:
    U(r) = r^2 * ln(r) for r > 0, and U(0) = 0.
    """
    r_safe = np.where(r > 1e-12, r, 1.0)
    u = (r_safe ** 2) * np.log(r_safe)
    return np.where(r > 1e-12, u, 0.0)


def tps_kernel_3d(r: np.ndarray) -> np.ndarray:
    """
    Radial basis function kernel for 3D Thin Plate Spline:
    U(r) = |r| (standard biharmonic/polyharmonic in 3D, Tang et al. 2019).
    """
    return np.abs(r)


def pairwise_euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes pairwise Euclidean distances between points in x and y."""
    diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


class ThinPlateSpline:
    """
    Closed-form Thin Plate Spline (TPS) Interpolator and Non-Rigid Warper.
    Supports both 2D (landmarks) and 3D coordinate spaces with optional L2 smoothing regularization (alpha).
    """

    def __init__(self, alpha: float = 0.001, dim: int = 2):
        """
        Parameters
        ----------
        alpha : float
            Regularization parameter (alpha >= 0). 
            alpha=0 guarantees exact interpolation at control points.
            alpha > 0 introduces smoothing and minimizes bending energy, preventing extreme shearing.
        dim : int
            Coordinate dimensionality (2 or 3). Default is 2.
        """
        self.alpha = float(alpha)
        self.dim = int(dim)
        self.source_points: Optional[np.ndarray] = None
        self.target_points: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None         # W (k x dim)
        self.affine_coeffs: Optional[np.ndarray] = None   # A ((dim + 1) x dim)
        self.kernel_matrix: Optional[np.ndarray] = None   # K (k x k)
        self.bending_energy: float = 0.0

    def _eval_kernel(self, r: np.ndarray) -> np.ndarray:
        if self.dim == 2:
            return tps_kernel_2d(r)
        elif self.dim == 3:
            return tps_kernel_3d(r)
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}. Expected 2 or 3.")

    def fit(self, source_points: np.ndarray, target_points: np.ndarray) -> "ThinPlateSpline":
        """
        Fits the TPS transformation mapping source_points -> target_points.

        Parameters
        ----------
        source_points : np.ndarray of shape (k, dim)
            Control points before deformation.
        target_points : np.ndarray of shape (k, dim)
            Control points after deformation.
        """
        source_points = np.asarray(source_points, dtype=np.float64)
        target_points = np.asarray(target_points, dtype=np.float64)

        k, d = source_points.shape
        if d != self.dim:
            raise ValueError(f"source_points dimension {d} does not match TPS dim {self.dim}")
        if target_points.shape != (k, d):
            raise ValueError(f"target_points shape {target_points.shape} must match source shape {(k, d)}")
        if k < self.dim + 1:
            raise ValueError(f"At least {self.dim + 1} control points are required to fit a {self.dim}D TPS.")

        self.source_points = source_points
        self.target_points = target_points

        # 1. Compute pairwise distance matrix between control points
        dist_matrix = pairwise_euclidean_distance(source_points, source_points)
        K = self._eval_kernel(dist_matrix)
        self.kernel_matrix = K

        # 2. Regularization along diagonal
        K_reg = K + self.alpha * np.eye(k)

        # 3. Construct affine design matrix P = [1, x1, x2, ...]
        P = np.hstack([np.ones((k, 1), dtype=np.float64), source_points])  # (k, d + 1)

        # 4. Assemble the partitioned block linear system L:
        # [ K + alpha * I    P  ] [ W ] = [ target_points ]
        # [      P^T         0  ] [ A ]   [       0       ]
        L_top = np.hstack([K_reg, P])
        L_bot = np.hstack([P.T, np.zeros((d + 1, d + 1), dtype=np.float64)])
        L = np.vstack([L_top, L_bot])  # (k + d + 1, k + d + 1)

        # Right hand side Y
        Y = np.vstack([target_points, np.zeros((d + 1, d), dtype=np.float64)])  # (k + d + 1, d)

        # Solve system via robust pseudo-inverse / least squares
        solution, residuals, rank, s = np.linalg.lstsq(L, Y, rcond=1e-12)

        self.weights = solution[:k, :]        # (k, d)
        self.affine_coeffs = solution[k:, :]  # (d + 1, d)

        # 5. Compute bending energy E_b = Tr(W^T * K * W)
        self.bending_energy = float(np.trace(self.weights.T @ K @ self.weights))

        return self

    def transform(self, points: np.ndarray) -> np.ndarray:
        """
        Applies the TPS mapping f(x) to arbitrary query points.

        Parameters
        ----------
        points : np.ndarray of shape (N, dim)

        Returns
        -------
        warped_points : np.ndarray of shape (N, dim)
        """
        if self.weights is None or self.affine_coeffs is None:
            raise RuntimeError("ThinPlateSpline must be fitted before calling transform().")

        points = np.asarray(points, dtype=np.float64)
        n, d = points.shape

        # Distances from query points to source control points
        dist_matrix = pairwise_euclidean_distance(points, self.source_points)  # (N, k)
        K_query = self._eval_kernel(dist_matrix)                               # (N, k)

        # Affine component: [1, x1, x2, ...] @ A
        P_query = np.hstack([np.ones((n, 1), dtype=np.float64), points])       # (N, d + 1)
        affine_part = P_query @ self.affine_coeffs                             # (N, d)

        # Non-affine bending component: K_query @ W
        warp_part = K_query @ self.weights                                     # (N, d)

        return affine_part + warp_part


def tps_warp_landmarks(
    source_landmarks: np.ndarray,
    target_control: np.ndarray,
    source_control: Optional[np.ndarray] = None,
    alpha: float = 0.005,
    dim: int = 2
) -> Tuple[np.ndarray, float]:
    """
    Convenience function: fits a TPS from source_control -> target_control
    and transforms source_landmarks. If source_control is None, source_landmarks is used as control.

    Returns:
        (warped_landmarks, bending_energy)
    """
    if source_control is None:
        source_control = source_landmarks

    tps = ThinPlateSpline(alpha=alpha, dim=dim)
    tps.fit(source_control, target_control)
    warped = tps.transform(source_landmarks)
    return warped, tps.bending_energy


def create_tps_deformation_grid(
    tps: ThinPlateSpline,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    grid_size: int = 25,
    margin_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a regular 2D mesh grid and warps it with the fitted TPS.
    Used for plotting deformation grids and visualizing bending fields.

    Returns:
        (X_grid, Y_grid, X_warped, Y_warped)
    """
    if tps.source_points is None:
        raise RuntimeError("TPS model is not fitted.")

    if bounds is None:
        min_x = np.min(tps.source_points[:, 0])
        max_x = np.max(tps.source_points[:, 0])
        min_y = np.min(tps.source_points[:, 1])
        max_y = np.max(tps.source_points[:, 1])

        dx = max_x - min_x
        dy = max_y - min_y
        min_x -= dx * margin_ratio
        max_x += dx * margin_ratio
        min_y -= dy * margin_ratio
        max_y += dy * margin_ratio
    else:
        min_x, max_x, min_y, max_y = bounds

    xs = np.linspace(min_x, max_x, grid_size)
    ys = np.linspace(min_y, max_y, grid_size)
    X_grid, Y_grid = np.meshgrid(xs, ys)

    grid_points = np.column_stack([X_grid.flatten(), Y_grid.flatten()])
    warped_points = tps.transform(grid_points)

    X_warped = warped_points[:, 0].reshape(grid_size, grid_size)
    Y_warped = warped_points[:, 1].reshape(grid_size, grid_size)

    return X_grid, Y_grid, X_warped, Y_warped
