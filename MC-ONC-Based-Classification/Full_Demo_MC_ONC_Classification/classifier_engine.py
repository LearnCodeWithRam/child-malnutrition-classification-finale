"""
classifier_engine.py
====================
Integrated Phase 2 Diagnosis Engine for Child Acute Malnutrition Screening:
- Tier 1: Frontline Offline Mobile Ensemble (40% Shrinkage LDA + 60% XGBoost) with 3-Tier Traffic Light Triage
- Tier 2: Online Cloud Deep Learning Graph Attention Network (EGAT) with 17 Nodes & 9 Transverse Calipers
- Multimodal Escalation Option: Definitively resolves borderline Yellow cases using shape + tape MUAC.
"""

import os
import io
import json
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modular_models import (
    ShrinkageLDAPredictor,
    XGBoostShapePredictor,
    CloudEGATPredictor,
    OfflineEnsemblePredictor,
    MultimodalResolver,
    ENABLE_CLOUD_EGAT,
    DEFAULT_PRIMARY_MODEL,
    DEFAULT_WEIGHT_LDA,
    DEFAULT_WEIGHT_XGB,
    DEFAULT_CUTOFF_YELLOW,
    DEFAULT_CUTOFF_RED
)

CURRENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = CURRENT_DIR / "saved_models"

# ===========================================================================
# 🎛️ ONE-LINE MASTER CONFIGURATION SWITCHES (Modify here to add/remove/toggle)
# ===========================================================================
ENABLE_EGAT_BACKUP: bool = ENABLE_CLOUD_EGAT   # <-- One line to toggle EGAT backup True/False
PRIMARY_MODEL: str = DEFAULT_PRIMARY_MODEL     # <-- Options: 'ensemble', 'lda', 'xgboost', 'egat'
# ===========================================================================

# 17 Active Contour Landmarks for EGAT
ACTIVE_LANDMARK_IDS_17 = [1, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20]
ID2IDX_17 = {lid: i for i, lid in enumerate(ACTIVE_LANDMARK_IDS_17)}
N_NODES_17 = len(ACTIVE_LANDMARK_IDS_17)

# 18 Active Contour Landmarks for Classical Tabular Models (includes L11 reconstructed)
ACTIVE_LANDMARK_IDS_18 = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# 9 Transverse Calipers
CALIPERS_9 = [
    (1, 4),    (12, 4),   (14, 13),
    (16, 15),  (18, 17),  (10, 9),
    (8, 7),    (6, 5),    (20, 19)
]

CALIPER_NAMES_9 = [
    "Proximal Shoulder Closure (L1-L4)",
    "Proximal Deltoid (L12-L4)",
    "Proximal Brachium/Bicep (L14-L13)",
    "Mid-Upper Arm / MUAC Site (L16-L15)",
    "Distal Brachium / Pre-Elbow (L18-L17)",
    "Elbow Joint Epicondyles (L10-L9)",
    "Proximal Forearm (L8-L7)",
    "Mid Forearm (L6-L5)",
    "Distal Wrist Closure (L20-L19)"
]


# ---------------------------------------------------------------------------
# Integrated Classification Engine
# ---------------------------------------------------------------------------
class MalnutritionClassificationEngine:
    def __init__(self, models_dir: Union[str, Path] = None,
                 enable_egat_backup: bool = ENABLE_EGAT_BACKUP,
                 primary_model: str = PRIMARY_MODEL):
        if models_dir is None:
            models_dir = MODELS_DIR
        self.models_dir = Path(models_dir)
        self.enable_egat_backup = enable_egat_backup
        self.primary_model = primary_model

        # 1. Load Morphometric Alignment Artifacts
        self.allom_reg = joblib.load(self.models_dir / "allometric_regression_model.joblib")
        self.ref_shape = np.load(self.models_dir / "procrustes_reference_shape.npy")  # (20, 2)
        self.mean_aligned = np.load(self.models_dir / "mean_aligned_landmarks.npy")  # (40,)

        # Precompute residual extraction indices for 18 active landmarks
        self.residual_indices_18 = []
        for lid in ACTIVE_LANDMARK_IDS_18:
            idx = lid - 1
            self.residual_indices_18.extend([idx * 2, idx * 2 + 1])

        # Load Ensemble configuration & cutoffs
        config_path = self.models_dir / "ensemble_config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.ensemble_config = json.load(f)
        else:
            self.ensemble_config = {}

        weights = self.ensemble_config.get("ensemble_weights", {})
        self.weight_lda = float(weights.get("shrinkage_lda", DEFAULT_WEIGHT_LDA))
        self.weight_xgb = float(weights.get("xgboost_shape", DEFAULT_WEIGHT_XGB))

        cutoffs = self.ensemble_config.get("traffic_light_cutoffs", {})
        self.cutoff_red_sam = float(cutoffs.get("red_critical_lower", DEFAULT_CUTOFF_RED))
        self.cutoff_yellow_mam = float(cutoffs.get("green_normal_upper", DEFAULT_CUTOFF_YELLOW))

        # 2. Instantiate Loosely Coupled Modular Model Predictors
        self.lda_predictor = ShrinkageLDAPredictor(self.models_dir)
        self.xgb_predictor = XGBoostShapePredictor(self.models_dir)
        self.egat_predictor = CloudEGATPredictor(self.models_dir) if self.enable_egat_backup else None

        # 3. Instantiate Grouped Offline Ensemble Predictor
        self.ensemble_predictor = OfflineEnsemblePredictor(
            lda_predictor=self.lda_predictor,
            xgb_predictor=self.xgb_predictor,
            weight_lda=self.weight_lda,
            weight_xgb=self.weight_xgb,
            cutoff_yellow=self.cutoff_yellow_mam,
            cutoff_red=self.cutoff_red_sam
        )

        # 4. Instantiate Multimodal Resolver
        self.multimodal_resolver = MultimodalResolver(self.models_dir)

    def _parse_to_20x2(self, raw_input: Union[Dict, List, np.ndarray]) -> np.ndarray:
        """Parses landmark inputs into standardized (20, 2) array."""
        if isinstance(raw_input, dict):
            coords = np.zeros((20, 2), dtype=float)
            for i in range(1, 21):
                kx = f"X{i}" if f"X{i}" in raw_input else f"x{i}"
                ky = f"Y{i}" if f"Y{i}" in raw_input else f"y{i}"
                coords[i - 1, 0] = float(raw_input.get(kx, 0.0))
                coords[i - 1, 1] = float(raw_input.get(ky, 0.0))
            if np.allclose(coords[10], [0.0, 0.0]) or np.allclose(coords[10], coords[3]):
                coords[10] = (coords[9] + coords[11]) / 2.0
            return coords

        elif isinstance(raw_input, list):
            # If list of landmark dicts from contour_extractor
            coords_dict = {}
            for item in raw_input:
                if isinstance(item, dict) and "Landmark_ID" in item:
                    lid = item["Landmark_ID"]
                    coords_dict[f"X{lid}"] = item.get("X_cartesian_cm", item.get("X_cm", 0.0))
                    coords_dict[f"Y{lid}"] = item.get("Y_cartesian_cm", item.get("Y_cm", 0.0))
            return self._parse_to_20x2(coords_dict)

        arr = np.array(raw_input, dtype=float)
        if arr.shape == (22, 2):
            coords = arr[:20].copy()
            if np.allclose(coords[10], [0.0, 0.0]) or np.allclose(coords[10], coords[3]):
                coords[10] = (coords[9] + coords[11]) / 2.0
            return coords
        elif arr.shape == (20, 2):
            coords = arr.copy()
            if np.allclose(coords[10], [0.0, 0.0]) or np.allclose(coords[10], coords[3]):
                coords[10] = (coords[9] + coords[11]) / 2.0
            return coords
        else:
            raise ValueError(f"Unrecognized coordinate array shape: {arr.shape}")

    def align_and_extract_residuals(self, raw_input: Union[Dict, List, np.ndarray]) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Runs Generalized Procrustes Analysis (fGPA) and allometric size decoupling.
        Returns:
            residuals_36 (36D vector)
            centroid_size (float)
            sh_aligned (20, 2)
            sh_res_20 (20, 2)
        """
        sh = self._parse_to_20x2(raw_input)

        # 1. Centering to origin
        centroid = np.mean(sh, axis=0)
        sh_centered = sh - centroid

        # 2. Scaling by Centroid Size
        centroid_size = max(float(np.sqrt(np.sum(sh_centered ** 2))), 1e-6)
        sh_scaled = sh_centered / centroid_size

        # 3. Procrustes Rotational Superimposition via SVD
        H = sh_scaled.T @ self.ref_shape
        U, S, Vt = np.linalg.svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        sh_aligned = sh_scaled @ R

        # 4. Allometric Ontogenetic Size Decoupling on ln(CS)
        aligned_40 = sh_aligned.flatten()
        log_cs = np.array([[np.log(centroid_size)]])
        allom_pred = self.allom_reg.predict(log_cs)[0]
        residuals_40 = aligned_40 - allom_pred + self.mean_aligned

        residuals_36 = residuals_40[self.residual_indices_18]
        sh_res_20 = residuals_40.reshape((20, 2))

        return residuals_36, centroid_size, sh_aligned, sh_res_20

    def evaluate_cloud_egat(self, sh_aligned_20: np.ndarray, sh_res_20: np.ndarray, centroid_size: float) -> Dict:
        """Evaluates Deep Learning 17-Node EGAT model with 9 transverse calipers."""
        # Extract 17 active contour nodes
        nodes_idx_17 = [lid - 1 for lid in ACTIVE_LANDMARK_IDS_17]
        pos_17 = sh_aligned_20[nodes_idx_17]  # (17, 2)
        res_17 = sh_res_20[nodes_idx_17]      # (17, 2)

        # Assemble 8D Node Features
        r_polar = np.sqrt(pos_17[:, 0]**2 + pos_17[:, 1]**2)
        theta_polar = np.arctan2(pos_17[:, 1], pos_17[:, 0])
        log_cs_val = np.log(centroid_size)
        log_cs_norm = float((log_cs_val - 3.70) / 0.16)
        lid_norm = np.array(ACTIVE_LANDMARK_IDS_17, dtype=np.float32) / 20.0

        node_feats = np.column_stack([
            pos_17,
            res_17,
            r_polar,
            theta_polar,
            np.full(N_NODES_17, log_cs_norm, dtype=np.float32),
            lid_norm
        ]).astype(np.float32)

        # Assemble 8D Edge Features (65 directed edges)
        senders_np = self.senders.numpy()
        receivers_np = self.receivers.numpy()
        pos_u = pos_17[senders_np]
        pos_v = pos_17[receivers_np]
        disp = pos_v - pos_u
        dists = np.linalg.norm(disp, axis=1, keepdims=True)
        strain = (dists.squeeze() - self.template_edge_dists) / self.template_edge_dists

        onehot = np.zeros((len(senders_np), 4), dtype=np.float32)
        for i, t in enumerate(self.edge_types_arr):
            onehot[i, t] = 1.0

        edge_feats = np.column_stack([
            dists, disp, strain[:, None], onehot
        ]).astype(np.float32)

        # Convert to PyTorch tensors (Batch size = 1)
        h_t = torch.tensor(node_feats, dtype=torch.float32).unsqueeze(0)
        e_t = torch.tensor(edge_feats, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits, alpha = self.egat_model(h_t, e_t, self.senders, self.receivers)
            probs = F.softmax(logits, dim=-1).squeeze().numpy()
            p_egat = float(probs[1])
            attn_mean = alpha.squeeze(0).mean(dim=-1).numpy()  # (65,)

        # Extract 9 Caliper Attention Weights
        caliper_attn = []
        for idx, (u, v) in enumerate(CALIPERS_9):
            edge_idx = self.caliper_directed_indices[idx]
            weight = float(attn_mean[edge_idx])
            caliper_attn.append({
                "caliper_id": idx + 1,
                "name": CALIPER_NAMES_9[idx],
                "pair": f"L{u} - L{v}",
                "attention_weight": round(weight, 4),
                "is_muac_site": (u == 16 and v == 15)
            })

        # Identify Primary Wasting Locus (highest attention caliper)
        top_caliper = max(caliper_attn, key=lambda c: c["attention_weight"])

        # Render Wireframe Attention Heatmap
        fig_heat, ax_heat = plt.subplots(figsize=(9, 4.5), facecolor='#ffffff')
        ax_heat.set_facecolor('#0f172a')  # Dark sleek background for heatmaps

        # Upper and lower contours
        upper_pts = [ID2IDX_17[lid] for lid in [1, 12, 14, 16, 18, 10, 8, 6, 20]]
        lower_pts = [ID2IDX_17[lid] for lid in [4, 13, 15, 17, 9, 7, 5, 19]]

        ax_heat.plot(pos_17[upper_pts, 0], pos_17[upper_pts, 1], color='#38bdf8', lw=2.0, alpha=0.7, label='Upper Contour')
        ax_heat.plot(pos_17[lower_pts, 0], pos_17[lower_pts, 1], color='#38bdf8', lw=2.0, alpha=0.7, label='Lower Contour')

        # Calipers colored by attention magnitude
        weights_arr = np.array([c["attention_weight"] for c in caliper_attn])
        norm_weights = (weights_arr - weights_arr.min()) / (weights_arr.max() - weights_arr.min() + 1e-6)
        cmap = plt.cm.plasma

        for idx, (u, v) in enumerate(CALIPERS_9):
            ui, vi = ID2IDX_17[u], ID2IDX_17[v]
            color = cmap(norm_weights[idx])
            lw = 2.0 + norm_weights[idx] * 3.5
            ax_heat.plot([pos_17[ui, 0], pos_17[vi, 0]], [pos_17[ui, 1], pos_17[vi, 1]],
                         color=color, lw=lw, alpha=0.9, zorder=3)

        # Plot nodes
        ax_heat.scatter(pos_17[:, 0], pos_17[:, 1], color='#ffffff', s=45, zorder=5)

        for i, lid in enumerate(ACTIVE_LANDMARK_IDS_17):
            ax_heat.annotate(f"{lid}", (pos_17[i, 0], pos_17[i, 1]),
                             xytext=(0, 6), textcoords='offset points', fontsize=8, color='#f1f5f9', ha='center')

        # Title and badges
        ax_heat.set_title(f"Tier 2 Online EGAT 9-Caliper Attention Heatmap (P_EGAT = {p_egat:.3f})\n"
                          f"Primary Wasting Locus: {top_caliper['name']}",
                          fontsize=10.5, fontweight='bold', color='#f8fafc')
        ax_heat.set_aspect('equal')
        ax_heat.axis('off')
        plt.tight_layout()

        buf = io.BytesIO()
        fig_heat.savefig(buf, format='png', dpi=140, bbox_inches='tight', facecolor='#0f172a')
        buf.seek(0)
        heatmap_b64 = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        plt.close(fig_heat)

        if p_egat >= 0.50:
            tier = "RED"
            badge = "🔴 RED: High Risk / Acute Malnutrition (SAM/MAM)"
            action = "Urgent clinical assessment and nutrition rehabilitation referral."
            category = "High Risk Malnourished (Acute Wasting Detected)"
        elif p_egat >= 0.25:
            tier = "YELLOW"
            badge = "🟡 YELLOW: Moderate Risk / Borderline Morphometric Deformation"
            action = "Community health monitoring, dietary intervention, and follow-up screening."
            category = "Borderline / Early Mild Wasting"
        else:
            tier = "GREEN"
            badge = "🟢 GREEN: Optimal Nutritional Status (Normal / Healthy ONC)"
            action = "Child is thriving normally. Continue standard routine monthly growth monitoring."
            category = "Optimal Healthy (Normal Arm Contour)"

        return {
            "probability_egat": round(p_egat, 4),
            "triage_tier": tier,
            "clinical_badge": badge,
            "clinical_action": action,
            "nutritional_category": category,
            "primary_wasting_site": top_caliper["name"],
            "calipers_attention": caliper_attn,
            "attention_heatmap_base64": heatmap_b64
        }

    def evaluate_offline_ensemble(self, residuals_36: np.ndarray, centroid_size: float = 0.0) -> Dict:
        """Evaluates Tier 1 Primary Offline Mobile Ensemble via modular OfflineEnsemblePredictor."""
        return self.ensemble_predictor.diagnose(residuals_36, centroid_size)

    def run_multimodal_confirmation(self, residuals_36: np.ndarray, gender: str, tape_muac_cm: float) -> Dict:
        """Definitively resolves borderline Yellow cases using modular 38D MultimodalResolver."""
        return self.multimodal_resolver.diagnose(residuals_36, gender, tape_muac_cm)

    def diagnose_full_pipeline(self, raw_landmarks: Union[Dict, List, np.ndarray],
                               gender: str = "M",
                               tape_muac_cm: Optional[float] = None,
                               enable_egat: Optional[bool] = None,
                               primary_model: Optional[str] = None) -> Dict:
        """
        Executes Modular Multi-Model Diagnosis:
        - Standalone Individual Models: Shrinkage LDA, XGBoost Shape, Cloud EGAT
        - Grouped Ensemble Model: 40% LDA + 60% XGBoost Soft Blend (3-Tier Traffic Light)
        - Backup Verification: Tier 2 Online EGAT (runs in parallel if enable_egat=True)
        - Secondary Escalation: Multimodal Model if tape MUAC is provided for borderline Yellow cases

        ONE-LINE SWITCH:
        Pass enable_egat=False (or set ENABLE_EGAT_BACKUP = False) to disable EGAT with zero graph overhead!
        """
        if enable_egat is None:
            enable_egat = self.enable_egat_backup
        if primary_model is None:
            primary_model = self.primary_model

        residuals_36, cs, sh_aligned_20, sh_res_20 = self.align_and_extract_residuals(raw_landmarks)

        # 1. Evaluate Individual Standalone Models
        lda_eval = self.lda_predictor.diagnose(residuals_36, cutoff_yellow=self.cutoff_yellow_mam, cutoff_red=self.cutoff_red_sam)
        xgb_eval = self.xgb_predictor.diagnose(residuals_36, cutoff_yellow=self.cutoff_yellow_mam, cutoff_red=self.cutoff_red_sam)

        # 2. Evaluate Grouped Ensemble Model
        ensemble_eval = self.ensemble_predictor.diagnose(residuals_36, centroid_size=cs)

        # 3. Evaluate Backup EGAT (Deep Learning Graph Attention)
        if enable_egat:
            if self.egat_predictor is None:
                self.egat_predictor = CloudEGATPredictor(self.models_dir)
            egat_eval = self.egat_predictor.diagnose(sh_aligned_20, sh_res_20, cs)
        else:
            egat_eval = {
                "status": "disabled",
                "model_name": "Cloud EGAT (Backup)",
                "message": "Cloud EGAT backup disabled via one-line toggle (ENABLE_EGAT_BACKUP = False)."
            }

        # 4. Select Primary Triage Output based on primary_model switch
        if primary_model == "lda":
            primary_triage = lda_eval
        elif primary_model == "xgboost":
            primary_triage = xgb_eval
        elif primary_model == "egat" and enable_egat:
            primary_triage = egat_eval
        else:
            primary_triage = ensemble_eval

        # 5. Optional Multimodal Confirmation for Borderline Yellow cases
        multimodal_res = None
        if tape_muac_cm is not None and tape_muac_cm > 0:
            multimodal_res = self.multimodal_resolver.diagnose(residuals_36, gender, tape_muac_cm)

        # 6. Model Agreement Check
        egat_tier = egat_eval.get("triage_tier") if isinstance(egat_eval, dict) else None
        models_agree = (primary_triage.get("triage_tier") == egat_tier) if (enable_egat and egat_tier) else True

        return {
            "status": "success",
            "primary_model_name": primary_triage.get("model_name", primary_model),
            "primary_triage": primary_triage,
            "individual_models": {
                "shrinkage_lda": lda_eval,
                "xgboost_shape": xgb_eval,
                "cloud_egat": egat_eval
            },
            "grouped_models": {
                "offline_ensemble_lda_xgboost": ensemble_eval
            },
            "cloud_egat_backup": egat_eval,
            "models_in_agreement": models_agree,
            "multimodal_confirmation": multimodal_res,
            "offline_ensemble": ensemble_eval,   # backwards compatibility
            "cloud_egat": egat_eval,              # backwards compatibility
            "centroid_size": round(float(cs), 2),
            "residuals_36": [round(float(v), 5) for v in residuals_36]
        }


