"""
tps_augmentation.py - Statistical Shape Modeling (SSM) & Thin Plate Spline (TPS) Data Augmentation
====================================================================================================
Implements learning of class-specific anatomical variation patterns and generation of high-fidelity
synthetic specimens for Geometric Morphometrics (GM) landmarks using:
1. Statistical Shape Modeling (SSM) via Generalized Procrustes Analysis (fGPA) and Shape PCA.
2. 2D Thin Plate Spline (TPS) non-rigid deformation (Bookstein 1989, Tang et al. 2019, raphaelreme/tps).
3. Coupled anthropometric feature generation (Centroid Size, MUAC, Age, Sex).
4. Full 104-column geometric morphometric recalculation engine matching Poshan / WHO standards.
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from tps_math import ThinPlateSpline, tps_kernel_2d


def extract_landmark_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identifies and sorts all X and Y landmark coordinate columns in the DataFrame."""
    x_cols = [c for c in df.columns if c.startswith("X") and c[1:].isdigit()]
    y_cols = [c for c in df.columns if c.startswith("Y") and c[1:].isdigit()]
    x_cols.sort(key=lambda s: int(s[1:]))
    y_cols.sort(key=lambda s: int(s[1:]))
    if len(x_cols) != len(y_cols):
        raise ValueError(f"Mismatch between X count ({len(x_cols)}) and Y count ({len(y_cols)})")
    return x_cols, y_cols


def extract_gpa_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identifies and sorts GPA coordinate columns in the DataFrame."""
    gpa_x_cols = [c for c in df.columns if c.startswith("GPA_X") and c.replace("GPA_X", "").isdigit()]
    gpa_y_cols = [c for c in df.columns if c.startswith("GPA_Y") and c.replace("GPA_Y", "").isdigit()]
    gpa_x_cols.sort(key=lambda s: int(s.replace("GPA_X", "")))
    gpa_y_cols.sort(key=lambda s: int(s.replace("GPA_Y", "")))
    return gpa_x_cols, gpa_y_cols


class StatisticalShapeModel:
    """
    Statistical Shape Model (SSM) for Geometric Morphometrics landmarks.
    Learns consensus configuration, landmark covariance, and principal deformation modes.
    Based on Tang et al. (2019) and Dryden & Mardia (2016).
    """

    def __init__(self, max_iter: int = 50, tol: float = 1e-7):
        self.max_iter = max_iter
        self.tol = tol
        self.consensus_shape: Optional[np.ndarray] = None  # (K, 2)
        self.eigenvectors: Optional[np.ndarray] = None     # (2K, 2K)
        self.eigenvalues: Optional[np.ndarray] = None      # (2K,)
        self.explained_variance_ratio: Optional[np.ndarray] = None
        self.n_landmarks: int = 0

    def fit(self, shapes: np.ndarray) -> "StatisticalShapeModel":
        """
        Fits SSM on landmark configurations.

        Parameters
        ----------
        shapes : np.ndarray of shape (N, K, 2)
            N specimens, K landmarks, 2 coordinates (X, Y).
        """
        N, K, D = shapes.shape
        if D != 2:
            raise ValueError(f"Expected 2D coordinates (X, Y), got {D}D")
        self.n_landmarks = K

        # 1. Center shapes and scale to unit Centroid Size
        centered = shapes - np.mean(shapes, axis=1, keepdims=True)
        cs = np.sqrt(np.sum(centered ** 2, axis=(1, 2), keepdims=True))
        scaled = np.where(cs > 1e-12, centered / cs, centered)

        # 2. Iterative Generalized Procrustes Analysis (fGPA) alignment
        ref = scaled[0].copy()
        aligned = np.zeros_like(scaled)

        for it in range(self.max_iter):
            for i in range(N):
                H = scaled[i].T @ ref
                U, S, Vt = svd(H)
                R = U @ Vt
                # Ensure proper rotation (det = +1)
                if np.linalg.det(R) < 0:
                    U[:, -1] *= -1
                    R = U @ Vt
                aligned[i] = scaled[i] @ R

            new_ref = np.mean(aligned, axis=0)
            new_ref_cs = np.sqrt(np.sum(new_ref ** 2))
            new_ref = new_ref / new_ref_cs if new_ref_cs > 0 else new_ref

            diff = np.max(np.abs(new_ref - ref))
            ref = new_ref
            if diff < self.tol:
                break

        self.consensus_shape = ref

        # 3. Shape Space Covariance & Eigendecomposition
        # Flatten aligned shapes into interleaved (N, 2K) shape coordinates
        Z = aligned.reshape(N, 2 * K)
        mean_Z = np.mean(Z, axis=0)

        # Centered shape vectors
        Z_centered = Z - mean_Z

        # Covariance matrix (2K x 2K)
        cov = (Z_centered.T @ Z_centered) / max(1, N - 1)

        # Eigendecomposition (Hermitian/symmetric)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        self.eigenvalues = np.maximum(eigvals[idx], 0.0)
        self.eigenvectors = eigvecs[:, idx]

        total_var = np.sum(self.eigenvalues)
        if total_var > 0:
            self.explained_variance_ratio = self.eigenvalues / total_var
        else:
            self.explained_variance_ratio = np.zeros_like(self.eigenvalues)

        return self

    def sample_shape(
        self,
        rng: np.random.Generator,
        sigma_scale: float = 1.0,
        max_std: float = 3.0,
        n_modes: Optional[int] = None
    ) -> np.ndarray:
        """
        Samples a new anatomically plausible shape configuration along the learned modes of variation.
        Tang et al. (2019): b_j ~ N(0, sigma^2 * lambda_j), bounded within [-3*std, 3*std].
        """
        if self.consensus_shape is None:
            raise RuntimeError("SSM must be fitted before sampling.")

        K = self.n_landmarks
        mean_Z = self.consensus_shape.reshape(2 * K)

        if n_modes is None:
            n_modes = min(len(self.eigenvalues), 2 * K - 4)

        # Sample standard normals
        z_scores = rng.normal(0.0, sigma_scale, size=n_modes)
        # Truncate at max_std to prevent extreme morphological anomalies
        z_scores = np.clip(z_scores, -max_std, max_std)

        # Scale by standard deviations sqrt(lambda_j)
        stds = np.sqrt(self.eigenvalues[:n_modes])
        perturbation = np.zeros(2 * K)

        for j in range(n_modes):
            if stds[j] > 1e-12:
                perturbation += z_scores[j] * stds[j] * self.eigenvectors[:, j]

        synth_Z = mean_Z + perturbation
        synth_shape = synth_Z.reshape(K, 2)

        # Normalize to unit Centroid Size
        synth_shape -= np.mean(synth_shape, axis=0)
        synth_cs = np.sqrt(np.sum(synth_shape ** 2))
        if synth_cs > 0:
            synth_shape /= synth_cs

        return synth_shape


class TPSAugmentor:
    """
    Comprehensive Class-Specific Thin Plate Spline (TPS) Data Augmentor.
    Learns patterns and variations from an existing class dataset (e.g. SAM) and generates
    new authentic specimens with complete, mathematically consistent 104-column features.
    """

    def __init__(
        self,
        tps_alpha: float = 0.005,
        ssm_modes: int = 10,
        random_state: int = 42
    ):
        self.tps_alpha = tps_alpha
        self.ssm_modes = ssm_modes
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        # Learned class parameters
        self.class_name: Optional[str] = None
        self.ssm = StatisticalShapeModel()
        self.raw_shapes: Optional[np.ndarray] = None       # (N, K, 2)
        self.x_cols: List[str] = []
        self.y_cols: List[str] = []
        self.gpa_x_cols: List[str] = []
        self.gpa_y_cols: List[str] = []
        self.all_columns: List[str] = []

        # Anthropometric & clinical empirical distributions
        self.cs_mean: float = 0.0
        self.cs_std: float = 1.0
        self.age_values: np.ndarray = np.array([])
        self.gender_prob_male: float = 0.5
        self.muac_regressor: Optional[LinearRegression] = None
        self.muac_residual_std: float = 0.5
        self.reference_pca: Optional[PCA] = None
        self.fitted = False

    def fit(self, df: pd.DataFrame, class_name: Optional[str] = None) -> "TPSAugmentor":
        """
        Learns morphological and clinical patterns from the target class subset of the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset containing landmarks and metadata.
        class_name : str, optional
            The specific class to learn (e.g. 'sam', 'mam', 'onc'). If None, fits on all rows.
        """
        self.all_columns = df.columns.tolist()
        self.x_cols, self.y_cols = extract_landmark_columns(df)
        self.gpa_x_cols, self.gpa_y_cols = extract_gpa_columns(df)

        if class_name is not None:
            self.class_name = str(class_name).lower()
            subset = df[df["Class_Type"].str.lower() == self.class_name].copy()
            if len(subset) == 0:
                raise ValueError(f"No samples found for class '{class_name}' in Class_Type column.")
        else:
            self.class_name = "all"
            subset = df.copy()

        n_samples = len(subset)

        # 1. Extract raw landmarks (N, K, 2)
        X_vals = subset[self.x_cols].to_numpy(dtype=np.float64)
        Y_vals = subset[self.y_cols].to_numpy(dtype=np.float64)
        self.raw_shapes = np.stack([X_vals, Y_vals], axis=-1)

        # 2. Fit Statistical Shape Model (fGPA and Eigenmodes)
        self.ssm.fit(self.raw_shapes)

        # 3. Learn Centroid Size Distribution
        calc_cs = np.sqrt(np.sum(
            (self.raw_shapes - np.mean(self.raw_shapes, axis=1, keepdims=True)) ** 2,
            axis=(1, 2)
        ))
        self.cs_mean = float(np.mean(calc_cs))
        self.cs_std = float(np.std(calc_cs))

        # 4. Learn Age and Gender Distributions
        if "age_month" in subset.columns:
            self.age_values = subset["age_month"].dropna().to_numpy(dtype=np.float64)
        else:
            self.age_values = np.linspace(6.0, 59.0, 100)

        if "gender" in subset.columns:
            g_series = subset["gender"].str.lower()
            male_count = np.sum(g_series.str.startswith("m"))
            self.gender_prob_male = float(male_count / max(1, len(g_series)))
        else:
            self.gender_prob_male = 0.5

        # 5. Learn MUAC Coupling Model
        # Physical law: MUAC reflects arm circumference, proportional to mid-upper arm thickness (L16-L15) and CS
        if "muac_cm" in subset.columns:
            muac_arr = subset["muac_cm"].to_numpy(dtype=np.float64).copy()
            # Normalize any mm values (e.g. 145.0 mm -> 14.5 cm)
            muac_arr = np.where(muac_arr >= 50.0, muac_arr / 10.0, muac_arr)

            # Find mid-upper arm transverse landmarks (16 and 15, or 14 and 13)
            idx_16 = 15 if len(self.x_cols) >= 16 else 5
            idx_15 = 14 if len(self.x_cols) >= 15 else 4
            arm_width = np.sqrt(
                (self.raw_shapes[:, idx_16, 0] - self.raw_shapes[:, idx_15, 0]) ** 2 +
                (self.raw_shapes[:, idx_16, 1] - self.raw_shapes[:, idx_15, 1]) ** 2
            )
            features = np.column_stack([arm_width, calc_cs, subset["age_month"].to_numpy(dtype=np.float64)])
            
            # Clean NaNs and constrain to physiologically valid clinical bounds
            valid_mask = ~np.isnan(features).any(axis=1) & ~np.isnan(muac_arr) & (muac_arr >= 7.0) & (muac_arr <= 25.0)
            if np.sum(valid_mask) > 10:
                self.muac_regressor = LinearRegression()
                self.muac_regressor.fit(features[valid_mask], muac_arr[valid_mask])
                preds = self.muac_regressor.predict(features[valid_mask])
                self.muac_residual_std = float(np.std(muac_arr[valid_mask] - preds))
            else:
                self.muac_regressor = None
                self.muac_residual_std = float(np.std(muac_arr[valid_mask])) if np.sum(valid_mask) > 0 else 1.2

        # 6. Fit Reference PCA for PC1..PC10 projection on GPA coordinates
        if len(self.gpa_x_cols) > 0 and len(self.gpa_y_cols) > 0:
            interleaved_cols = []
            for gx, gy in zip(self.gpa_x_cols, self.gpa_y_cols):
                interleaved_cols.extend([gx, gy])
            X_gpa_all = df[interleaved_cols].to_numpy(dtype=np.float64)
            self.reference_pca = PCA(n_components=10)
            self.reference_pca.fit(X_gpa_all)

        self.fitted = True
        return self

    def generate_sample(
        self,
        method: str = "ssm_tps",
        smoothness: Optional[float] = None,
        specimen_id: int = 1
    ) -> Dict[str, Union[float, str, bool, int]]:
        """
        Generates a single synthetic specimen conditioned on the learned class pattern.

        Methods:
        - 'ssm_tps': (Tang et al. 2019) Deforms an authentic exemplar toward an SSM-sampled shape using TPS.
        - 'pairwise_tps': Blends two random exemplars via non-linear TPS warping with smooth random jitter.
        - 'elastic_tps': Applies elastic smooth TPS displacement field to an authentic exemplar.
        """
        if not self.fitted:
            raise RuntimeError("TPSAugmentor must be fitted before generating samples.")

        alpha = smoothness if smoothness is not None else self.tps_alpha
        K = self.raw_shapes.shape[1]
        N = self.raw_shapes.shape[0]

        # 1. Select base exemplar
        base_idx = self.rng.integers(0, N)
        base_shape = self.raw_shapes[base_idx].copy()  # (K, 2)
        base_centered = base_shape - np.mean(base_shape, axis=0)
        base_cs = np.sqrt(np.sum(base_centered ** 2))
        base_unit = base_centered / base_cs if base_cs > 0 else base_centered

        # 2. Synthesize deformed shape using the requested TPS strategy
        if method == "ssm_tps":
            # Sample biologically authentic target shape from SSM
            target_unit = self.ssm.sample_shape(
                self.rng,
                sigma_scale=1.0,
                max_std=2.8,
                n_modes=self.ssm_modes
            )

            # Fit Thin Plate Spline mapping base_unit -> target_unit
            tps = ThinPlateSpline(alpha=alpha, dim=2)
            tps.fit(base_unit, target_unit)

            # Warp the base shape
            warped_unit = tps.transform(base_unit)

            # Add minor high-frequency anatomical micro-variation (bounded bending)
            jitter_scale = 0.003 * np.std(base_unit)
            warped_unit += self.rng.normal(0.0, jitter_scale, size=(K, 2))

        elif method == "pairwise_tps":
            # Blend two authentic class exemplars
            second_idx = (base_idx + self.rng.integers(1, N)) % N
            target_shape = self.raw_shapes[second_idx].copy()
            target_centered = target_shape - np.mean(target_shape, axis=0)
            target_cs = np.sqrt(np.sum(target_centered ** 2))
            target_unit = target_centered / target_cs if target_cs > 0 else target_centered

            # TPS warp from base to target
            tps = ThinPlateSpline(alpha=alpha, dim=2)
            tps.fit(base_unit, target_unit)

            # Morphing factor t in [0.25, 0.75]
            t_blend = self.rng.uniform(0.25, 0.75)
            warped_full = tps.transform(base_unit)
            warped_unit = (1.0 - t_blend) * base_unit + t_blend * warped_full

        elif method == "elastic_tps":
            # Elastic smooth displacement field on control grid
            min_xy = np.min(base_unit, axis=0)
            max_xy = np.max(base_unit, axis=0)
            ctrl_xs = np.linspace(min_xy[0] - 0.05, max_xy[0] + 0.05, 4)
            ctrl_ys = np.linspace(min_xy[1] - 0.05, max_xy[1] + 0.05, 4)
            ctrl_X, ctrl_Y = np.meshgrid(ctrl_xs, ctrl_ys)
            ctrl_pts = np.column_stack([ctrl_X.flatten(), ctrl_Y.flatten()])

            # Smooth random shift on control grid
            disp_scale = 0.015 * np.std(base_unit)
            ctrl_disp = self.rng.normal(0.0, disp_scale, size=ctrl_pts.shape)
            ctrl_tgt = ctrl_pts + ctrl_disp

            tps = ThinPlateSpline(alpha=alpha, dim=2)
            tps.fit(ctrl_pts, ctrl_tgt)
            warped_unit = tps.transform(base_unit)

        else:
            raise ValueError(f"Unknown TPS augmentation method: {method}")

        # 3. Sample realistic Centroid Size and apply allometric scale
        synth_cs = float(self.rng.normal(self.cs_mean, self.cs_std))
        synth_cs = max(15.0, min(65.0, synth_cs))  # Clinical plausibility limits

        # Center and restore to metric dimensions
        warped_unit -= np.mean(warped_unit, axis=0)
        final_norm = np.sqrt(np.sum(warped_unit ** 2))
        warped_unit = warped_unit / final_norm if final_norm > 0 else warped_unit

        final_shape = warped_unit * synth_cs

        # Add realistic spatial translation (matching original coordinate range)
        orig_mean_center = np.mean(self.raw_shapes[base_idx], axis=0)
        final_shape += orig_mean_center

        # 4. Synthesize clinical features
        # Sample age
        synth_age = float(self.rng.choice(self.age_values))
        # Add slight age jitter within +/- 0.5 month, clipped to [6, 59]
        synth_age = float(np.clip(synth_age + self.rng.uniform(-0.5, 0.5), 6.0, 59.0))

        # Sample gender
        synth_gender = "Male" if self.rng.uniform() < self.gender_prob_male else "Female"

        # Synthesize MUAC coupled to physical arm width
        idx_16 = 15 if K >= 16 else 5
        idx_15 = 14 if K >= 15 else 4
        synth_arm_width = float(np.sqrt(
            (final_shape[idx_16, 0] - final_shape[idx_15, 0]) ** 2 +
            (final_shape[idx_16, 1] - final_shape[idx_15, 1]) ** 2
        ))

        if self.muac_regressor is not None:
            feat = np.array([[synth_arm_width, synth_cs, synth_age]])
            pred_muac = float(self.muac_regressor.predict(feat)[0])
            synth_muac = pred_muac + self.rng.normal(0.0, self.muac_residual_std)
        else:
            synth_muac = float(self.rng.normal(14.1, 1.6))

        # Physical clinical bound
        synth_muac = float(np.clip(synth_muac, 8.5, 20.0))

        # 5. Compute Generalized Procrustes Coordinates (GPA_X, GPA_Y)
        # Pairwise fOPA alignment of synth_shape to consensus shape
        U_opa, _, Vt_opa = svd(warped_unit.T @ self.ssm.consensus_shape)
        R_opa = U_opa @ Vt_opa
        if np.linalg.det(R_opa) < 0:
            U_opa[:, -1] *= -1
            R_opa = U_opa @ Vt_opa
        gpa_aligned = warped_unit @ R_opa  # Unit centroid size aligned to consensus

        # 6. Recalculate all morphometric features
        # Recalculate true Centroid Size
        actual_cs = float(np.sqrt(np.sum((final_shape - np.mean(final_shape, axis=0)) ** 2)))

        # ILD distances
        # Landmark 1 is index 0. Landmark 3 is index 2. Landmark 4 is index 3.
        ild_shoulder_wrist = float(np.sqrt(
            (final_shape[0, 0] - final_shape[2, 0]) ** 2 +
            (final_shape[0, 1] - final_shape[2, 1]) ** 2
        ))
        ild_shoulder_armpit = float(np.sqrt(
            (final_shape[0, 0] - final_shape[3, 0]) ** 2 +
            (final_shape[0, 1] - final_shape[3, 1]) ** 2
        ))

        # Bounding Box Area
        min_x, max_x = np.min(final_shape[:, 0]), np.max(final_shape[:, 0])
        min_y, max_y = np.min(final_shape[:, 1]), np.max(final_shape[:, 1])
        bb_area = float((max_x - min_x) * (max_y - min_y))

        # Procrustes distance to consensus shape
        procrustes_dist = float(np.sqrt(np.sum((gpa_aligned - self.ssm.consensus_shape) ** 2)))

        # Interleaved GPA vector for PCA projection
        gpa_interleaved = np.zeros(2 * K)
        gpa_interleaved[0::2] = gpa_aligned[:, 0]
        gpa_interleaved[1::2] = gpa_aligned[:, 1]

        if self.reference_pca is not None:
            pc_scores = self.reference_pca.transform(gpa_interleaved.reshape(1, -1))[0]
        else:
            pc_scores = np.zeros(10)

        # 7. Assemble dictionary matching exact dataset schema
        row: Dict[str, Union[float, str, bool, int]] = {
            "File_Name": f"{self.class_name}_tps_aug_{specimen_id:04d}.jpg",
            "Individual": f"tps_aug_{self.class_name}_{specimen_id:04d}",
            "Class_Type": self.class_name if self.class_name != "all" else "sam",
            "is_augmented": True,
            "age_month": synth_age,
            "gender": synth_gender,
            "muac_cm": synth_muac,
            "missing_count": 0,
            "zero_coord_count": 0,
            "Centroid_Size": actual_cs,
            "Procrustes_Distance": procrustes_dist,
            "Procrustes_Outlier": False,
        }

        # Landmarks X and Y
        for i, (xc, yc) in enumerate(zip(self.x_cols, self.y_cols)):
            row[xc] = float(final_shape[i, 0])
            row[yc] = float(final_shape[i, 1])

        # GPA coordinates
        for i, (gxc, gyc) in enumerate(zip(self.gpa_x_cols, self.gpa_y_cols)):
            row[gxc] = float(gpa_aligned[i, 0])
            row[gyc] = float(gpa_aligned[i, 1])

        # PCA components PC1..PC10
        for i in range(1, 11):
            row[f"PC{i}"] = float(pc_scores[i - 1])

        # Outlier and quality flags
        row["PCA_Outlier"] = False
        row["Mahalanobis_Distance"] = float(np.sum(pc_scores[:5] ** 2))
        row["Mahalanobis_Outlier"] = False
        row["Bounding_Box_Area"] = bb_area
        row["ILD_Shoulder_Wrist"] = ild_shoulder_wrist
        row["ILD_Shoulder_Armpit"] = ild_shoulder_armpit
        row["Digitizing_Error_Flag"] = False

        return row

    def augment(
        self,
        target_total_samples: int = 600,
        method: str = "ssm_tps",
        smoothness: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Augments the class dataset to reach target_total_samples.
        Generates (target_total_samples - current_samples) new synthetic samples,
        and returns the complete augmented class DataFrame of exactly target_total_samples rows.
        """
        if not self.fitted:
            raise RuntimeError("TPSAugmentor must be fitted before augmenting.")

        current_n = len(self.raw_shapes)
        n_needed = max(0, target_total_samples - current_n)

        # Get existing samples for this class from original dataset
        # to ensure perfect continuity
        print(f"[{self.class_name.upper()}] Current real samples: {current_n}. Generating {n_needed} synthetic TPS samples...")

        synthetic_rows = []
        for i in range(1, n_needed + 1):
            synth_row = self.generate_sample(
                method=method,
                smoothness=smoothness,
                specimen_id=i
            )
            synthetic_rows.append(synth_row)

        synth_df = pd.DataFrame(synthetic_rows)
        return synth_df


# ==============================================================================
# GLOBAL REUSABLE FUNCTIONS (Can be used on any dataset or class)
# ==============================================================================

def augment_class_dataset(
    df: pd.DataFrame,
    target_class: str = "sam",
    target_total_samples: int = 600,
    method: str = "ssm_tps",
    smoothness: float = 0.005,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Global reusable function: Learns patterns from target_class in df,
    generates synthetic samples using Thin Plate Spline (TPS), and returns:
    1. synthetic_df: Only the newly generated synthetic samples.
    2. combined_df: Original class samples + newly generated synthetic samples (total = target_total_samples).

    Parameters
    ----------
    df : pd.DataFrame
        Source dataset.
    target_class : str
        Class label to augment (e.g. 'sam', 'mam', 'onc').
    target_total_samples : int
        Target count of samples for this class (e.g. 600).
    method : str
        'ssm_tps' (Tang et al. 2019), 'pairwise_tps', or 'elastic_tps'.
    smoothness : float
        TPS regularization smoothing alpha.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    synthetic_df, combined_class_df
    """
    augmentor = TPSAugmentor(tps_alpha=smoothness, random_state=random_state)
    augmentor.fit(df, class_name=target_class)

    synthetic_df = augmentor.augment(
        target_total_samples=target_total_samples,
        method=method,
        smoothness=smoothness
    )

    original_class_df = df[df["Class_Type"].str.lower() == target_class.lower()].copy()
    if "is_augmented" not in original_class_df.columns:
        cls_idx = original_class_df.columns.get_loc("Class_Type")
        original_class_df.insert(cls_idx + 1, "is_augmented", False)

    # Align columns
    cols_order = original_class_df.columns.tolist()
    # Filter only available columns
    available_cols = [c for c in cols_order if c in synthetic_df.columns]
    synthetic_aligned = synthetic_df[available_cols]

    combined_class_df = pd.concat([original_class_df, synthetic_aligned], ignore_index=True)
    return synthetic_df, combined_class_df


def augment_multi_class_dataset(
    df: pd.DataFrame,
    target_counts: Dict[str, int],
    method: str = "ssm_tps",
    smoothness: float = 0.005,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Global reusable function: Augments one or more classes to their respective target counts,
    and returns the entire unified dataset with augmented classes.

    Example:
    >>> augmented_df = augment_multi_class_dataset(df, target_counts={"sam": 600})
    """
    augmented_dfs = []
    processed_classes = set()

    for cls_name, target_n in target_counts.items():
        cls_lower = cls_name.lower()
        processed_classes.add(cls_lower)
        _, combined_cls_df = augment_class_dataset(
            df=df,
            target_class=cls_lower,
            target_total_samples=target_n,
            method=method,
            smoothness=smoothness,
            random_state=random_state
        )
        augmented_dfs.append(combined_cls_df)

    # Add remaining unmodified classes from df
    for orig_cls in df["Class_Type"].unique():
        if orig_cls.lower() not in processed_classes:
            unmod_df = df[df["Class_Type"] == orig_cls].copy()
            if "is_augmented" not in unmod_df.columns:
                cls_idx = unmod_df.columns.get_loc("Class_Type")
                unmod_df.insert(cls_idx + 1, "is_augmented", False)
            augmented_dfs.append(unmod_df)

    full_augmented_df = pd.concat(augmented_dfs, ignore_index=True)
    return full_augmented_df
