"""
tps_model_training.py - Global Methodology for Geometric Morphometric Classification
=====================================================================================
A generic, publication-grade training and evaluation pipeline for Geometric Morphometrics
shape landmarks based on Medialdea et al. (Nature Scientific Reports 2025, 15:3906).

Methodology Workflow:
1. Cohort Preprocessing & Class Balance
2. 20-Landmark Extraction & Reconstruction (Forearm Midpoint Landmark 11)
3. Full Generalized Procrustes Analysis (fGPA) Shape Alignment
4. Allometric Size Regression on log(Centroid Size)
5. Collinearity Elimination (18 Landmarks / 36 Variables vs. 20 Landmarks / 40 Variables)
6. Multimodal Feature Fusion (Shape Residuals + Clinical MUAC)
7. Multi-Classifier Training (LDA, Logistic Regression, kNN, Random Forest, XGBoost)
8. Cross-Validation (Stratified K-Fold and LOOCV)
9. Production Parameter & Template Export
"""

import json
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any

from scipy.linalg import svd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_validate
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, recall_score,
    precision_score, f1_score, roc_auc_score, confusion_matrix
)


def extract_ordered_landmarks_20(df: pd.DataFrame) -> Tuple[np.ndarray, List[int]]:
    """
    Extracts all 20 anatomical landmarks in order 1..20.
    Reconstructs Landmark 11 as forearm midpoint between L10 and L12 if missing.
    
    Returns:
        shapes: np.ndarray of shape (N, 20, 2)
        landmark_ids: list of landmark integers [1..20]
    """
    df_work = df.copy()

    # Reconstruct L11 if not present
    if "X11" not in df_work.columns or "Y11" not in df_work.columns:
        if "X10" in df_work.columns and "X12" in df_work.columns:
            df_work["X11"] = (df_work["X10"].astype(float) + df_work["X12"].astype(float)) / 2.0
            df_work["Y11"] = (df_work["Y10"].astype(float) + df_work["Y12"].astype(float)) / 2.0
        else:
            raise ValueError("Cannot reconstruct Landmark 11: missing X10 or X12")

    landmark_ids = list(range(1, 21))
    N = len(df_work)
    shapes = np.zeros((N, 20, 2), dtype=np.float64)

    for idx, lid in enumerate(landmark_ids):
        shapes[:, idx, 0] = df_work[f"X{lid}"].to_numpy(dtype=np.float64)
        shapes[:, idx, 1] = df_work[f"Y{lid}"].to_numpy(dtype=np.float64)

    return shapes, landmark_ids


def run_fgpa_alignment(shapes: np.ndarray, max_iter: int = 50, tol: float = 1e-7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes Full Generalized Procrustes Analysis (fGPA) on landmark shapes.

    Returns:
        shapes_gpa: np.ndarray of shape (N, K, 2)
        consensus: np.ndarray of shape (K, 2)
        centroid_sizes: np.ndarray of shape (N,)
    """
    N, K, D = shapes.shape

    # 1. Center shapes
    centroids = np.mean(shapes, axis=1, keepdims=True)
    centered = shapes - centroids

    # 2. Centroid Sizes CS = sqrt(sum(x^2 + y^2))
    cs = np.sqrt(np.sum(centered ** 2, axis=(1, 2)))
    scaled = np.where(cs[:, np.newaxis, np.newaxis] > 1e-12, centered / cs[:, np.newaxis, np.newaxis], centered)

    # 3. Iterative alignment to consensus
    ref = scaled[0].copy()
    aligned = np.zeros_like(scaled)

    for _ in range(max_iter):
        for i in range(N):
            H = scaled[i].T @ ref
            U, S, Vt = svd(H)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt
            aligned[i] = scaled[i] @ R

        new_ref = np.mean(aligned, axis=0)
        norm_ref = np.sqrt(np.sum(new_ref ** 2))
        new_ref = new_ref / norm_ref if norm_ref > 0 else new_ref

        if np.max(np.abs(new_ref - ref)) < tol:
            break
        ref = new_ref

    return aligned, ref, cs


def compute_allometric_residuals(shapes_gpa: np.ndarray, centroid_sizes: np.ndarray) -> Tuple[np.ndarray, LinearRegression]:
    """
    Performs multivariate linear regression on log(Centroid Size) to eliminate
    growth/allometric size confounding from shape coordinates.

    Returns:
        residuals: np.ndarray of shape (N, 2 * K)
        regressor: fitted LinearRegression model
    """
    N, K, D = shapes_gpa.shape
    X_interleaved = shapes_gpa.reshape(N, K * D)
    log_cs = np.log(np.maximum(centroid_sizes, 1e-6)).reshape(-1, 1)

    reg = LinearRegression()
    reg.fit(log_cs, X_interleaved)
    predicted = reg.predict(log_cs)
    residuals = X_interleaved - predicted

    return residuals, reg


def drop_collinear_elbow_landmarks(features_40: np.ndarray, landmark_ids_20: List[int]) -> np.ndarray:
    """
    Eliminates semilandmarks 2 and 3 along the elbow axis to resolve extreme collinearity
    (condition number reduction from ~10^16 to ~1.3*10^3 as proven by Medialdea et al.).
    Reduces from 40 variables (20 landmarks) to 36 variables (18 landmarks).
    """
    kept_lids = [lid for lid in landmark_ids_20 if lid not in (2, 3)]
    kept_indices = []
    for lid in kept_lids:
        orig_idx = landmark_ids_20.index(lid)
        kept_indices.extend([orig_idx * 2, orig_idx * 2 + 1])

    return features_40[:, kept_indices]


class MorphometricsClassifierPipeline:
    """
    End-to-End Geometric Morphometrics Classifier Pipeline.
    Can be trained and evaluated on any class combination (e.g. SAM vs. ONC or SAM vs. MAM vs. ONC).
    """

    def __init__(
        self,
        use_allometry: bool = True,
        use_18_landmarks: bool = True,
        include_muac: bool = True,
        random_state: int = 42
    ):
        self.use_allometry = use_allometry
        self.use_18_landmarks = use_18_landmarks
        self.include_muac = include_muac
        self.random_state = random_state

        self.consensus_template: Optional[np.ndarray] = None
        self.allometric_regressor: Optional[LinearRegression] = None
        self.landmark_ids: List[int] = list(range(1, 21))
        self.feature_names: List[str] = []
        self.classes: List[str] = []

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts landmarks, executes fGPA, allometric correction, collinearity drop,
        and optional multimodal MUAC concatenation.

        Returns:
            X: np.ndarray of shape (N, n_features)
            y: np.ndarray of shape (N,)
            cs: np.ndarray of shape (N,)
        """
        shapes, self.landmark_ids = extract_ordered_landmarks_20(df)
        aligned, consensus, cs = run_fgpa_alignment(shapes)
        self.consensus_template = consensus

        N, K, _ = aligned.shape
        X_40 = aligned.reshape(N, 2 * K)

        if self.use_allometry:
            residuals_40, reg = compute_allometric_residuals(aligned, cs)
            self.allometric_regressor = reg
            X_base = residuals_40
        else:
            X_base = X_40

        if self.use_18_landmarks:
            X_shape = drop_collinear_elbow_landmarks(X_base, self.landmark_ids)
            base_names = [f"Res_L{lid}_{ax}" for lid in self.landmark_ids if lid not in (2, 3) for ax in ("X", "Y")]
        else:
            X_shape = X_base
            base_names = [f"Res_L{lid}_{ax}" for lid in self.landmark_ids for ax in ("X", "Y")]

        if self.include_muac and "muac_cm" in df.columns:
            muac_vals = df["muac_cm"].to_numpy(dtype=np.float64).reshape(-1, 1).copy()
            # Normalize any mm values (e.g. 145.0 mm -> 14.5 cm)
            muac_vals = np.where(muac_vals >= 50.0, muac_vals / 10.0, muac_vals)
            X = np.hstack([X_shape, muac_vals])
            self.feature_names = base_names + ["muac_cm"]
        else:
            X = X_shape
            self.feature_names = base_names

        y = df["Class_Type"].str.lower().to_numpy()
        return X, y, cs


def evaluate_models_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    target_class: str = "sam",
    n_splits: int = 5,
    run_loocv: bool = False,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluates standard machine learning classifiers across Stratified K-Fold CV
    and optional Leave-One-Out CV (LOOCV).

    Classifiers:
    - LDA (Linear Discriminant Analysis)
    - Logistic Regression (LR)
    - k-Nearest Neighbors (kNN, k=5)
    - Random Forest (RF)
    - XGBoost (XGB)
    """
    models = {
        "LDA": LinearDiscriminantAnalysis(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "kNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=random_state, eval_metric="logloss")
    }

    unique_classes = sorted(list(np.unique(y)))
    is_binary = len(unique_classes) == 2

    # Map labels to 0/1 if binary for ROC-AUC
    if is_binary:
        y_num = np.where(y == target_class.lower(), 1, 0)
    else:
        label_map = {cls: i for i, cls in enumerate(unique_classes)}
        y_num = np.array([label_map[c] for c in y])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []
    sens_key = f"Sensitivity ({target_class.upper()} Recall)"

    for name, clf in models.items():
        accs, b_accs, recs, specs, precs, f1s, aucs = [], [], [], [], [], [], []

        for train_idx, test_idx in cv.split(X, y_num):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_num[train_idx], y_num[test_idx]

            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)

            accs.append(accuracy_score(y_test, preds))
            b_accs.append(balanced_accuracy_score(y_test, preds))
            f1s.append(f1_score(y_test, preds, average="binary" if is_binary else "macro", zero_division=0))

            if is_binary:
                recs.append(recall_score(y_test, preds, zero_division=0))  # Sensitivity for target_class
                precs.append(precision_score(y_test, preds, zero_division=0))
                cm = confusion_matrix(y_test, preds, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                specs.append(spec)

                if hasattr(clf, "predict_proba"):
                    try:
                        probs = clf.predict_proba(X_test)[:, 1]
                        aucs.append(roc_auc_score(y_test, probs))
                    except Exception:
                        aucs.append(np.nan)
                else:
                    aucs.append(np.nan)

        res_row = {
            "Classifier": name,
            "CV_Folds": n_splits,
            "Accuracy": float(np.mean(accs)),
            "Balanced_Accuracy": float(np.mean(b_accs)),
            sens_key: float(np.mean(recs)) if is_binary else float(np.mean(b_accs)),
            "Sensitivity (Recall)": float(np.mean(recs)) if is_binary else float(np.mean(b_accs)),
            "Specificity": float(np.mean(specs)) if is_binary else np.nan,
            "Precision": float(np.mean(precs)) if is_binary else np.nan,
            "F1_Score": float(np.mean(f1s)),
            "ROC_AUC": float(np.nanmean(aucs)) if (is_binary and not np.all(np.isnan(aucs))) else np.nan,
        }
        results.append(res_row)

    # Optional LOOCV for LDA
    if run_loocv and len(X) <= 1200:
        loo = LeaveOneOut()
        lda = LinearDiscriminantAnalysis()
        loo_preds = []
        loo_trues = []

        for train_idx, test_idx in loo.split(X, y_num):
            lda.fit(X[train_idx], y_num[train_idx])
            pred = lda.predict(X[test_idx])
            loo_preds.append(pred[0])
            loo_trues.append(y_num[test_idx][0])

        loo_acc = accuracy_score(loo_trues, loo_preds)
        if is_binary:
            loo_sens = recall_score(loo_trues, loo_preds, zero_division=0)
            cm = confusion_matrix(loo_trues, loo_preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            loo_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            loo_sens = balanced_accuracy_score(loo_trues, loo_preds)
            loo_spec = np.nan

        results.append({
            "Classifier": "LDA (LOOCV Empirical Benchmark)",
            "CV_Folds": "LOOCV",
            "Accuracy": float(loo_acc),
            "Balanced_Accuracy": float(balanced_accuracy_score(loo_trues, loo_preds)),
            sens_key: float(loo_sens),
            "Sensitivity (Recall)": float(loo_sens),
            "Specificity": float(loo_spec),
            "Precision": float(precision_score(loo_trues, loo_preds, zero_division=0)) if is_binary else np.nan,
            "F1_Score": float(f1_score(loo_trues, loo_preds, average="binary" if is_binary else "macro", zero_division=0)),
            "ROC_AUC": np.nan
        })

    return pd.DataFrame(results)


def train_evaluate_morphometrics_pipeline(
    df: pd.DataFrame,
    class_column: str = "Class_Type",
    classes_to_include: Optional[List[str]] = None,
    target_class: str = "sam",
    balance_classes: bool = True,
    use_allometry: bool = True,
    use_18_landmarks: bool = True,
    include_muac: bool = True,
    cv_folds: int = 5,
    run_loocv: bool = False,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Global methodology to train and evaluate classification models on any class or dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing landmark coordinates and class labels.
    class_column : str
        Name of class column (default 'Class_Type').
    classes_to_include : list of str, optional
        Subset of classes to classify (e.g. ['sam', 'onc'] or ['sam', 'mam', 'onc']).
        If None, all classes in class_column are included.
    target_class : str
        Target positive class for clinical sensitivity calculation (default 'sam').
    balance_classes : bool
        If True, downsamples the larger classes to match the target class count in this cohort.
    use_allometry : bool
        Whether to perform size-correction regression on log(Centroid Size).
    use_18_landmarks : bool
        Whether to drop semilandmarks 2 and 3 to eliminate collinearity.
    include_muac : bool
        Whether to concatenate clinical MUAC (cm) for multimodal fusion.
    cv_folds : int
        Number of Stratified K-Fold splits (default 5).
    run_loocv : bool
        Whether to run Leave-One-Out Cross-Validation for LDA.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    results_df : pd.DataFrame
        Table of performance metrics for all classifiers.
    pipeline_artifacts : dict
        Fitted models, consensus template, feature names, and metadata.
    """
    df_cohort = df.copy()
    df_cohort[class_column] = df_cohort[class_column].str.lower()

    if classes_to_include is not None:
        classes_lower = [c.lower() for c in classes_to_include]
        df_cohort = df_cohort[df_cohort[class_column].isin(classes_lower)].copy()

    # Filter clinical age [6, 59] months if available
    if "age_month" in df_cohort.columns:
        df_cohort = df_cohort[(df_cohort["age_month"] >= 6.0) & (df_cohort["age_month"] <= 59.0)].copy()

    # Class balancing if requested
    if balance_classes:
        min_count = df_cohort[class_column].value_counts().min()
        balanced_dfs = []
        for cls in df_cohort[class_column].unique():
            cls_df = df_cohort[df_cohort[class_column] == cls]
            if len(cls_df) > min_count:
                sampled = cls_df.sample(n=min_count, random_state=random_state)
            else:
                sampled = cls_df
            balanced_dfs.append(sampled)
        df_cohort = pd.concat(balanced_dfs, ignore_index=True)

    print(f"Cohort cohort breakdown (N = {len(df_cohort)}):")
    print(df_cohort[class_column].value_counts())

    pipeline = MorphometricsClassifierPipeline(
        use_allometry=use_allometry,
        use_18_landmarks=use_18_landmarks,
        include_muac=include_muac,
        random_state=random_state
    )

    X, y, cs = pipeline.prepare_features(df_cohort)

    results_df = evaluate_models_cross_validation(
        X=X,
        y=y,
        target_class=target_class,
        n_splits=cv_folds,
        run_loocv=run_loocv,
        random_state=random_state
    )

    # Train final production LDA and XGBoost on entire cohort
    lda_prod = LinearDiscriminantAnalysis()
    lda_prod.fit(X, y)

    xgb_prod = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=random_state)
    label_map = {cls: i for i, cls in enumerate(np.unique(y))}
    xgb_prod.fit(X, np.array([label_map[c] for c in y]))

    artifacts = {
        "classes": list(np.unique(y)),
        "n_samples": len(df_cohort),
        "n_features": X.shape[1],
        "feature_names": pipeline.feature_names,
        "consensus_template": pipeline.consensus_template.tolist() if pipeline.consensus_template is not None else None,
        "production_lda": lda_prod,
        "production_xgb": xgb_prod,
        "label_mapping": label_map
    }

    return results_df, artifacts
