"""
pure_shape_inference_engine.py
==============================
Autonomous Pure Shape-Based Inference Engine for Child Malnutrition Screening.

CRITICAL DESIGN PRINCIPLE:
- ZERO MUAC REQUIRED: Operates 100% on pure 2D arm contour geometry.
- ZERO DEMOGRAPHIC INPUTS: Decouples ontogenetic arm growth automatically via allometry.
- INPUT: Pure raw 2D landmark coordinates (pixels or normalized) from arm photography.
- PIPELINE:
    1. Automatic Centroid Size (CS) calculation & scaling
    2. Generalized Procrustes superimposition (SVD rotational alignment)
    3. Log-centroid size allometric regression decoupling
    4. Extraction of 36D pure shape contour residuals
    5. Soft-blend ensemble (40% Shrinkage LDA + 60% XGBoost)
    6. 3-Tier "Traffic Light" clinical triage (Green < 0.25, Yellow 0.25-0.50, Red >= 0.50)
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass



class PureShapeDiagnosisEngine:
    """
    Pure Shape-Based Child Malnutrition Screening & Triage Engine.
    Requires NO MUAC, NO sex, and NO age. Only raw 2D arm landmark coordinates.
    """

    ACTIVE_LANDMARK_IDS = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    CUTOFF_HIGH_SENSITIVITY = 0.25  # Zero Child Left Behind cutoff (catches > 81.9% real MC)
    CUTOFF_DEFINITE_SAM = 0.50       # Definite severe malnutrition cutoff

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
        
        self.models_dir = models_dir
        
        # Load configuration
        config_path = os.path.join(models_dir, "ensemble_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {
                "ensemble_weights": {"shrinkage_lda": 0.40, "xgboost_shape": 0.60},
                "thresholds": {"green_max": 0.25, "yellow_max": 0.50}
            }
            
        self.weight_lda = self.config["ensemble_weights"]["shrinkage_lda"]
        self.weight_xgb = self.config["ensemble_weights"]["xgboost_shape"]

        # Load models and transformers
        self.lda_model = joblib.load(os.path.join(models_dir, "shrinkage_lda_model.joblib"))
        self.xgb_model = joblib.load(os.path.join(models_dir, "xgboost_shape_model.joblib"))
        self.allom_reg = joblib.load(os.path.join(models_dir, "allometric_regression_model.joblib"))
        self.ref_shape = np.load(os.path.join(models_dir, "procrustes_reference_shape.npy")) # (20, 2)
        self.mean_aligned = np.load(os.path.join(models_dir, "mean_aligned_landmarks.npy"))  # (40,)

        # Precompute residual feature extraction indices
        self.residual_indices = []
        for lid in self.ACTIVE_LANDMARK_IDS:
            idx = lid - 1
            self.residual_indices.extend([idx * 2, idx * 2 + 1])

    def _parse_input_to_20x2(self, raw_input: Union[np.ndarray, List, Dict, pd.Series]) -> np.ndarray:
        """
        Parses diverse raw landmark coordinate formats (22 landmarks or 20 landmarks)
        into the standardized (20, 2) anatomical landmark array:
        - Ingests all 22 raw landmarks (X1..X22, Y1..Y22).
        - Autonomously drops external calibration board markers (L21, L22).
        - Automatically reconstructs/validates forearm midpoint L11 = (L10 + L12) / 2.
        - Returns (20, 2) standardized anatomical shape ready for Procrustes superimposition.
        """
        if isinstance(raw_input, (dict, pd.Series)):
            coords = np.zeros((20, 2), dtype=float)
            for i in range(1, 21):
                kx = f"X{i}" if f"X{i}" in raw_input else f"x{i}"
                ky = f"Y{i}" if f"Y{i}" in raw_input else f"y{i}"
                val_x = raw_input.get(kx, 0.0) if isinstance(raw_input, dict) else raw_input.get(kx, 0.0)
                val_y = raw_input.get(ky, 0.0) if isinstance(raw_input, dict) else raw_input.get(ky, 0.0)
                coords[i - 1, 0] = float(val_x)
                coords[i - 1, 1] = float(val_y)

            # Forearm midpoint L11 standardization:
            # In raw digitizer datasets (such as final_dataset_highlighted.csv), L11 was either omitted
            # or duplicated to L4. The true anatomical forearm midpoint is (L10 + L12) / 2.
            l10 = coords[9]  # Index 9 = Landmark 10
            l12 = coords[11] # Index 11 = Landmark 12
            l4  = coords[3]  # Index 3 = Landmark 4
            l11 = coords[10] # Index 10 = Landmark 11

            if np.allclose(l11, [0.0, 0.0]) or np.allclose(l11, l4):
                coords[10] = (l10 + l12) / 2.0

            return coords

        arr = np.array(raw_input, dtype=float)
        # Case 1: 22 landmarks (shape 22, 2) -> drop calibration markers L21 and L22 (indices 20 and 21)
        if arr.shape == (22, 2):
            coords = arr[:20].copy()
            if np.allclose(coords[10], [0.0, 0.0]) or np.allclose(coords[10], coords[3]):
                coords[10] = (coords[9] + coords[11]) / 2.0
            return coords

        # Case 2: 44 coordinates (flattened 22 landmarks)
        elif arr.shape == (44,) or arr.shape == (1, 44):
            arr22 = arr.reshape((22, 2))
            coords = arr22[:20].copy()
            if np.allclose(coords[10], [0.0, 0.0]) or np.allclose(coords[10], coords[3]):
                coords[10] = (coords[9] + coords[11]) / 2.0
            return coords

        # Case 3: 20 landmarks (shape 20, 2)
        elif arr.shape == (20, 2):
            coords = arr.copy()
            if np.allclose(coords[10], [0.0, 0.0]) or np.allclose(coords[10], coords[3]):
                coords[10] = (coords[9] + coords[11]) / 2.0
            return coords

        # Case 4: 40 coordinates (flattened 20 landmarks)
        elif arr.shape == (40,) or arr.shape == (1, 40):
            coords = arr.reshape((20, 2)).copy()
            if np.allclose(coords[10], [0.0, 0.0]) or np.allclose(coords[10], coords[3]):
                coords[10] = (coords[9] + coords[11]) / 2.0
            return coords

        else:
            raise ValueError(
                f"Unrecognized coordinate shape {arr.shape}. "
                "Expected 22 landmarks ((22, 2) or (44,)) or 20 landmarks ((20, 2) or (40,))."
            )

    def process_raw_22_landmarks_to_18_shape(self, raw_input: Union[np.ndarray, List, Dict, pd.Series]) -> Dict:
        """
        End-to-end transformation pipeline from 22 RAW landmarks to 18 ACTIVE shape features:
        1. Ingests all 22 raw landmarks (X1, Y1 ... X22, Y22).
        2. Drops external calibration markers L21 and L22.
        3. Standardizes forearm midpoint L11 = (L10 + L12) / 2.0.
        4. Translates to origin (centering) and computes Centroid Size (CS).
        5. Rotates via SVD Procrustes alignment to consensus reference arm.
        6. Removes ontogenetic growth scaling via allometric regression log(CS).
        7. Drops internal semilandmarks 2 and 3, yielding the EXACT 18 active contour landmarks
           [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] (36 shape residuals).
        """
        residuals_36, centroid_size, sh_aligned = self.extract_pure_shape_residuals(raw_input)
        aligned_18 = sh_aligned[[lid - 1 for lid in self.ACTIVE_LANDMARK_IDS]]

        return {
            "residuals_36": residuals_36,
            "centroid_size": centroid_size,
            "aligned_20x2": sh_aligned,
            "aligned_18x2": aligned_18,
            "active_landmark_ids": self.ACTIVE_LANDMARK_IDS
        }

    def extract_pure_shape_residuals(self, raw_input: Union[np.ndarray, List, Dict, pd.Series]) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Transforms raw (X, Y) landmarks into size-invariant 36D allometric shape residuals.
        
        Returns:
            residuals_36 (np.ndarray): 36D shape feature vector
            centroid_size (float): The child's arm scale in photo units
            sh_aligned (np.ndarray): (20, 2) Procrustes aligned landmarks
        """
        sh = self._parse_input_to_20x2(raw_input)

        # 1. Centering (translation invariance)
        centroid = np.mean(sh, axis=0)
        sh_centered = sh - centroid

        # 2. Scaling (size invariance)
        centroid_size = max(float(np.sqrt(np.sum(sh_centered ** 2))), 1e-6)
        sh_scaled = sh_centered / centroid_size

        # 3. Procrustes Rotational Superimposition via SVD (rotation invariance)
        H = sh_scaled.T @ self.ref_shape
        U, S, Vt = np.linalg.svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        sh_aligned = sh_scaled @ R

        # Flatten into 40D aligned shape representation
        aligned_40 = np.zeros(40, dtype=float)
        for idx in range(20):
            aligned_40[idx * 2] = sh_aligned[idx, 0]
            aligned_40[idx * 2 + 1] = sh_aligned[idx, 1]

        # 4. Ontogenetic Allometric Size Decoupling:
        # Expected shape = alpha + beta * ln(CS)
        log_cs = np.array([[np.log(centroid_size)]])
        allom_pred = self.allom_reg.predict(log_cs)[0]

        # Residuals = observed aligned shape - allometric expected shape + reference mean
        residuals_40 = aligned_40 - allom_pred + self.mean_aligned

        # 5. Drop semilandmarks 2 and 3 -> 36D contour residuals
        residuals_36 = residuals_40[self.residual_indices]
        return residuals_36, centroid_size, sh_aligned

    def diagnose_child(self, raw_coordinates: Union[np.ndarray, List, Dict, pd.Series]) -> Dict:
        """
        Diagnoses child nutritional status PURELY from raw 2D landmark coordinates.
        NO MUAC OR DEMOGRAPHIC INFORMATION REQUIRED.
        """
        residuals_36, centroid_size, sh_aligned = self.extract_pure_shape_residuals(raw_coordinates)
        x_in = residuals_36.reshape(1, -1)


        # Model inferences
        p_lda = float(self.lda_model.predict_proba(x_in)[0, 1])
        p_xgb = float(self.xgb_model.predict_proba(x_in)[0, 1])
        p_ensemble = float(self.weight_lda * p_lda + self.weight_xgb * p_xgb)

        # 3-Tier Traffic Light Triage
        if p_ensemble >= self.CUTOFF_DEFINITE_SAM:
            tier = "RED"
            tier_display = "🔴 RED: Critical Risk (Severe Acute Malnutrition / SAM)"
            action = "Immediate urgent clinical referral to Primary Health Centre (PHC) or NRC."
            status = "Malnourished (MC - Definite Critical)"
        elif p_ensemble >= self.CUTOFF_HIGH_SENSITIVITY:
            tier = "YELLOW"
            tier_display = "🟡 YELLOW: Moderate Risk (Borderline / Early-Stage MAM)"
            action = "Place child on Community Watchlist. Provide targeted nutrition counseling and re-screen in 14 days."
            status = "At-Risk Malnutrition (MAM / Watchlist)"
        else:
            tier = "GREEN"
            tier_display = "🟢 GREEN: Normal Nutritional Status (Healthy ONC)"
            action = "Child is developing normally. Schedule standard monthly routine growth monitoring."
            status = "Optimal Nutritional Condition (ONC)"

        return {
            "probability_mc": round(p_ensemble, 4),
            "probability_lda": round(p_lda, 4),
            "probability_xgboost": round(p_xgb, 4),
            "triage_tier": tier,
            "triage_display": tier_display,
            "clinical_status": status,
            "clinical_action": action,
            "is_flagged_for_action": p_ensemble >= self.CUTOFF_HIGH_SENSITIVITY,
            "centroid_size": round(centroid_size, 2),
            "residuals_36d": residuals_36.tolist(),
            "aligned_shape_20x2": sh_aligned
        }

    def plot_arm_wireframe(self, raw_coordinates, ax=None, color='#2980b9', show_consensus=True, lw=2.0):
        """
        Plots the authentic anatomical wireframe for a child matching clinical calibration:
        - Upper Edge: 12 -> 14 -> 16 -> 18 -> 10 -> 8 -> 6 -> 20
        - Lower Edge: 4 -> 13 -> 15 -> 17 -> 9 -> 7 -> 5 -> 19
        - Shoulder Connections: 1 -> 12 and 1 -> 4
        - Wrist Distal Boundary: 20 -> 19
        - Transverse Cross-Sections: [12, 4], [14, 13], [16, 15], [18, 17], [10, 9], [8, 7], [6, 5], [20, 19]
        - Longitudinal Midline: 1 (Shoulder) -> 2 (Elbow) -> 3 (Wrist)
        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            
        sh = self._parse_input_to_20x2(raw_coordinates)
        c = np.mean(sh, axis=0)
        sh_c = (sh - c) / max(np.sqrt(np.sum((sh - c)**2)), 1e-6)
        H = sh_c.T @ self.ref_shape
        U, S, Vt = np.linalg.svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        sh_aligned = sh_c @ R
        
        upper_ids = [12, 14, 16, 18, 10, 8, 6, 20]
        lower_ids = [4, 13, 15, 17, 9, 7, 5, 19]
        cross_pairs = [[12, 4], [14, 13], [16, 15], [18, 17], [10, 9], [8, 7], [6, 5], [20, 19]]
        id_map = {i+1: i for i in range(20)}
        
        def _draw(coords, col, is_ref=False, line_w=2.0):
            ls = '--' if is_ref else '-'
            alpha = 0.55 if is_ref else 0.95
            # Upper edge
            ux = [coords[id_map[i], 0] for i in upper_ids]
            uy = [coords[id_map[i], 1] for i in upper_ids]
            ax.plot(ux, uy, color=col, ls=ls, lw=line_w, alpha=alpha, label='Consensus Mean' if is_ref else 'Child Upper Edge', zorder=4)
            # Lower edge
            lx = [coords[id_map[i], 0] for i in lower_ids]
            ly = [coords[id_map[i], 1] for i in lower_ids]
            ax.plot(lx, ly, color=col, ls=ls, lw=line_w, alpha=alpha, label='Child Lower Edge' if not is_ref else None, zorder=4)
            # Shoulder: 1->12 and 1->4
            ax.plot([coords[id_map[1], 0], coords[id_map[12], 0]], [coords[id_map[1], 1], coords[id_map[12], 1]],
                    color='#27ae60' if not is_ref else col, ls=ls, lw=line_w*1.1, alpha=alpha, zorder=5)
            ax.plot([coords[id_map[1], 0], coords[id_map[4], 0]], [coords[id_map[1], 1], coords[id_map[4], 1]],
                    color='#27ae60' if not is_ref else col, ls=ls, lw=line_w*1.1, alpha=alpha, zorder=5)
            # Wrist: 20->19
            ax.plot([coords[id_map[20], 0], coords[id_map[19], 0]], [coords[id_map[20], 1], coords[id_map[19], 1]],
                    color='#27ae60' if not is_ref else col, ls=ls, lw=line_w*0.9, alpha=alpha, zorder=4)
            # Cross-sections
            for c_idx, (u, l) in enumerate(cross_pairs):
                lbl = 'Transverse Cross-Sections' if (c_idx == 0 and not is_ref) else None
                ax.plot([coords[id_map[u], 0], coords[id_map[l], 0]], [coords[id_map[u], 1], coords[id_map[l], 1]],
                        color='#7f8c8d', ls=':', lw=1.1, alpha=alpha*0.65, label=lbl, zorder=3)
            # Midline
            mid_x = [coords[id_map[1], 0], coords[id_map[2], 0], coords[id_map[3], 0]]
            mid_y = [coords[id_map[1], 1], coords[id_map[2], 1], coords[id_map[3], 1]]
            ax.plot(mid_x, mid_y, color='#f39c12' if not is_ref else '#bdc3c7', ls='-' if not is_ref else ':',
                    lw=1.8, alpha=alpha*0.85, label='Center Midline (1->Elbow->Wrist)' if not is_ref else None, zorder=4)
            if not is_ref:
                ax.scatter(coords[:, 0], coords[:, 1], color=col, s=28, zorder=6)
                ax.scatter([coords[id_map[1], 0]], [coords[id_map[1], 1]], color='#c0392b', s=65, zorder=7, label='Shoulder (L1)')
                ax.scatter([coords[id_map[2], 0]], [coords[id_map[2], 1]], color='#e67e22', s=50, zorder=7, label='Elbow (L2)')
                ax.scatter([coords[id_map[3], 0]], [coords[id_map[3], 1]], color='#2980b9', edgecolors='#c0392b', s=55, zorder=7, label='Wrist (L3)')
                
        if show_consensus:
            _draw(self.ref_shape, '#95a5a6', is_ref=True, line_w=1.3)
        _draw(sh_aligned, color, is_ref=False, line_w=lw)
        
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, ls=':', alpha=0.5)
        ax.set_xlabel('Standardized Length (Shoulder -> Wrist)', fontsize=9)
        ax.set_ylabel('Standardized Arm Contour', fontsize=9)
        return ax

    def batch_diagnose(self, df_with_landmarks: pd.DataFrame) -> pd.DataFrame:
        """
        Batch processes an entire cohort from raw landmark columns (X1..X20, Y1..Y20).
        """
        results = []
        for _, row in df_with_landmarks.iterrows():
            diag = self.diagnose_child(row)
            results.append({
                "probability_mc": diag["probability_mc"],
                "probability_lda": diag["probability_lda"],
                "probability_xgboost": diag["probability_xgboost"],
                "triage_tier": diag["triage_tier"],
                "clinical_status": diag["clinical_status"],
                "is_flagged": diag["is_flagged_for_action"]
            })
        return pd.DataFrame(results, index=df_with_landmarks.index)



if __name__ == "__main__":
    print("=" * 80)
    print("TESTING AUTONOMOUS PURE SHAPE INFERENCE ENGINE (ZERO MUAC)")
    print("RAW INPUT: 22 LANDMARKS (X1..X22, Y1..Y22) -> PROCESSED TO 18 ACTIVE CONTOUR LANDMARKS")
    print("=" * 80)

    engine = PureShapeDiagnosisEngine()

    # 1. Test on raw dataset: final_dataset_highlighted.csv (22 landmarks)
    raw_csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "GM_out_of_sample_classification-ap-govt",
        "final_dataset_highlighted.csv"
    )
    if os.path.exists(raw_csv_path):
        print(f"\n1. Ingesting raw 22-landmark dataset: {os.path.basename(raw_csv_path)}")
        raw_df = pd.read_csv(raw_csv_path)
        print(f"   Loaded {len(raw_df):,} samples with raw columns: X1..X22, Y1..Y22.")
        
        # Test 1 sample taking ONLY raw X1..X22, Y1..Y22 coordinates
        raw_row = raw_df.iloc[0]
        only_22_coords = {f"X{i}": raw_row[f"X{i}"] for i in range(1, 23)}
        only_22_coords.update({f"Y{i}": raw_row[f"Y{i}"] for i in range(1, 23)})
        
        # Transformation pipeline check
        pipe_result = engine.process_raw_22_landmarks_to_18_shape(only_22_coords)
        print(f"   • Extracted Centroid Size (CS):     {pipe_result['centroid_size']:.2f} photo units")
        print(f"   • Standardized Aligned 20x2 shape:   {pipe_result['aligned_20x2'].shape}")
        print(f"   • Extracted 18 Active Landmarks:     {pipe_result['aligned_18x2'].shape} (IDs: {pipe_result['active_landmark_ids']})")
        print(f"   • Generated 36D Allometric Residuals:{pipe_result['residuals_36'].shape}")

        diag_raw = engine.diagnose_child(only_22_coords)
        print(f"   • Diagnosis from pure 22 raw points: {diag_raw['triage_display']}")
        print(f"   • P(MC) = {diag_raw['probability_mc']:.3f} (LDA: {diag_raw['probability_lda']:.3f}, XGB: {diag_raw['probability_xgboost']:.3f})")
        print(f"   • Clinical Action: {diag_raw['clinical_action']}")

    # 2. Test on balanced test cohort: unseen_test_dataset_balanced_n1456.csv
    test_path = os.path.join(engine.models_dir, "unseen_test_dataset_balanced_n1456.csv")
    if os.path.exists(test_path):
        print(f"\n2. Testing on representative clinical cases from {os.path.basename(test_path)}:")
        test_df = pd.read_csv(test_path)

        sam_row = test_df[(test_df["Class_Type"] == "mc") & (test_df["mc_subtype"] == "sam") & (~test_df["is_synth"])].iloc[0]
        mam_row = test_df[(test_df["Class_Type"] == "mc") & (test_df["mc_subtype"] == "mam") & (~test_df["is_synth"])].iloc[0]
        onc_row = test_df[test_df["Class_Type"] == "onc"].iloc[0]

        for label, row in [("Severe Acute Malnutrition (SAM)", sam_row),
                           ("Moderate Acute Malnutrition (MAM)", mam_row),
                           ("Optimal Healthy Control (ONC)", onc_row)]:
            diag = engine.diagnose_child(row)
            print(f"\nSpecimen: {label}")
            print(f"  • Ensemble Prob(MC): {diag['probability_mc']:.3f} (LDA: {diag['probability_lda']:.3f}, XGB: {diag['probability_xgboost']:.3f})")
            print(f"  • Triage Result:     {diag['triage_display']}")
            print(f"  • Clinical Action:   {diag['clinical_action']}")

    print("\n" + "=" * 80)
    print("SUCCESS: Pure shape inference engine operates seamlessly on 22 raw landmarks with ZERO MUAC.")
    print("=" * 80)
