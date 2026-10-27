"""
run_tps_augmentation_mc_onc.py - End-to-End Pipeline for MC Class TPS Augmentation (2X & ONC Parity)
====================================================================================================
Augments the Malnourished Condition (MC) cohort using Thin Plate Spline (TPS) non-rigid warping:
1. Stage 1 (2X MC Augmentation): Expands MC from 967 to 1,934 samples (967 real + 967 synthetic).
2. Stage 2 (ONC Parity Matching): Expands MC to 3,640 samples (967 real + 2,673 synthetic) matching ONC count.
3. Generates complete 104-column feature recalculations for all synthetic instances.
4. Trains and evaluates 5 standard ML paradigms (LDA, Logistic Regression, kNN, Random Forest, XGBoost)
   across Baseline (N=1,934), 2X Augmented (N=3,868), and Full Matched Parity (N=7,280) cohorts.
5. Produces publication-grade diagnostic wireframes, deformation grids, morphospace, and benchmark plots.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from tps_math import ThinPlateSpline, create_tps_deformation_grid
from tps_augmentation import (
    TPSAugmentor, StatisticalShapeModel,
    augment_class_dataset, augment_multi_class_dataset,
    extract_landmark_columns, extract_gpa_columns
)
from tps_model_training import (
    train_evaluate_morphometrics_pipeline,
    extract_ordered_landmarks_20, run_fgpa_alignment
)
from run_tps_augmentation_sam600 import plot_authentic_arm_wireframe


def run_mc_tps_augmentation():
    print("=" * 85)
    print("THIN PLATE SPLINE (TPS) DATA AUGMENTATION: MC CLASS (2X & ONC MATCHING)")
    print("=" * 85)

    # 1. Dataset Path
    csv_name = "final_cleaned_dataset_full_with_muac_and_class_corrected_with_who_R_package_and_removed_underweight_stunted_and_merge_SAM_MAM_to_MC_and_remove_borderline_DP.csv"
    csv_path = os.path.join(script_dir, csv_name)
    if not os.path.exists(csv_path):
        # Fallback to parent path
        csv_path = os.path.join(script_dir, "..", csv_name)

    print(f"\n[1] Ingesting clinical dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Class_Type distribution:")
    print(df["Class_Type"].value_counts())

    figures_dir = os.path.join(script_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    if "is_augmented" not in df.columns:
        cls_idx = df.columns.get_loc("Class_Type")
        df.insert(cls_idx + 1, "is_augmented", False)

    mc_real_df = df[df["Class_Type"].str.lower() == "mc"].copy()
    onc_real_df = df[df["Class_Type"].str.lower() == "onc"].copy()
    n_mc_real = len(mc_real_df)
    n_onc_real = len(onc_real_df)
    print(f"\nAuthentic Cohort Breakdown: MC = {n_mc_real}, ONC = {n_onc_real}")

    # 2. Fit Statistical Shape Model (SSM) on authentic MC instances
    print(f"\n[2] Fitting Statistical Shape Model (SSM) on authentic MC cohort (N = {n_mc_real})...")
    augmentor = TPSAugmentor(tps_alpha=0.005, ssm_modes=10, random_state=42)
    augmentor.fit(df, class_name="mc")

    cum_var = np.cumsum(augmentor.ssm.explained_variance_ratio)
    print(f"  - Fitted SSM Consensus Shape: {augmentor.ssm.n_landmarks} landmarks.")
    print(f"  - Mode 1 Variance Explained: {augmentor.ssm.explained_variance_ratio[0]*100:.2f}%")
    print(f"  - Top 3 Modes Cumulative Variance: {cum_var[2]*100:.2f}%")
    print(f"  - Top 5 Modes Cumulative Variance: {cum_var[4]*100:.2f}%")
    print(f"  - Top 10 Modes Cumulative Variance: {cum_var[min(9, len(cum_var)-1)]*100:.2f}%")
    print(f"  - Centroid Size: mean = {augmentor.cs_mean:.2f}, std = {augmentor.cs_std:.2f}")
    print(f"  - Clinical MUAC: residual std = {augmentor.muac_residual_std:.2f} cm")
    print(f"  - Biological Sex: {augmentor.gender_prob_male*100:.1f}% Male, {(1-augmentor.gender_prob_male)*100:.1f}% Female")

    # 3. Stage 1: 2X MC Augmentation (Target = 1,934 total MC: 967 real + 967 synthetic)
    print(f"\n[3] Executing Stage 1: 2X MC Augmentation (Target = {2 * n_mc_real} MC)...")
    synth_mc_967_df, mc_total_1934_df = augment_class_dataset(
        df=df,
        target_class="mc",
        target_total_samples=2 * n_mc_real,
        method="ssm_tps",
        smoothness=0.005,
        random_state=42
    )

    # 4. Stage 2: ONC Parity Matching (Target = 3,640 total MC: 967 real + 2,673 synthetic)
    print(f"\n[4] Executing Stage 2: ONC Parity Matching (Target = {n_onc_real} MC)...")
    synth_mc_2673_df, mc_total_3640_df = augment_class_dataset(
        df=df,
        target_class="mc",
        target_total_samples=n_onc_real,
        method="ssm_tps",
        smoothness=0.005,
        random_state=42
    )

    # Form balanced cohorts for benchmarking
    # Cohort 1: Baseline (967 MC vs 967 ONC)
    onc_sample_967 = onc_real_df.sample(n=n_mc_real, random_state=42)
    cohort_baseline = pd.concat([mc_real_df, onc_sample_967], ignore_index=True)

    # Cohort 2: 2X Balanced (1,934 MC vs 1,934 ONC)
    onc_sample_1934 = onc_real_df.sample(n=len(mc_total_1934_df), random_state=42)
    cohort_2x_balanced = pd.concat([mc_total_1934_df, onc_sample_1934], ignore_index=True)

    # Cohort 3: Full Parity Matched (3,640 MC vs 3,640 ONC)
    cohort_parity_matched = pd.concat([mc_total_3640_df, onc_real_df], ignore_index=True)

    # Save CSV files
    path_pure_2x = os.path.join(script_dir, "mc_pure_synthetic_2x_967.csv")
    path_total_2x = os.path.join(script_dir, "mc_augmented_2x_total_1934.csv")
    path_balanced_2x = os.path.join(script_dir, "final_cleaned_dataset_augmented_2x_mc1934_onc1934.csv")

    path_pure_parity = os.path.join(script_dir, "mc_pure_synthetic_onc_match_2673.csv")
    path_total_parity = os.path.join(script_dir, "mc_augmented_onc_match_total_3640.csv")
    path_balanced_parity = os.path.join(script_dir, "final_cleaned_dataset_augmented_parity_mc3640_onc3640.csv")

    synth_mc_967_df.to_csv(path_pure_2x, index=False)
    mc_total_1934_df.to_csv(path_total_2x, index=False)
    cohort_2x_balanced.to_csv(path_balanced_2x, index=False)

    synth_mc_2673_df.to_csv(path_pure_parity, index=False)
    mc_total_3640_df.to_csv(path_total_parity, index=False)
    cohort_parity_matched.to_csv(path_balanced_parity, index=False)

    print("\n[5] Saved Generated Datasets:")
    print(f"  -> {path_pure_2x} (N = {len(synth_mc_967_df)}, pure synthetic 2X MC)")
    print(f"  -> {path_total_2x} (N = {len(mc_total_1934_df)}, exactly 1,934 MC: 967 real + 967 synthetic)")
    print(f"  -> {path_balanced_2x} (N = {len(cohort_2x_balanced)}, balanced 2X: 1,934 MC vs 1,934 ONC)")
    print(f"  -> {path_pure_parity} (N = {len(synth_mc_2673_df)}, pure synthetic ONC match MC)")
    print(f"  -> {path_total_parity} (N = {len(mc_total_3640_df)}, exactly 3,640 MC: 967 real + 2,673 synthetic)")
    print(f"  -> {path_balanced_parity} (N = {len(cohort_parity_matched)}, full parity: 3,640 MC vs 3,640 ONC)")

    # 5. Model Benchmarking across 3 cohorts
    print("\n" + "=" * 85)
    print("[6] MULTI-CLASSIFIER BENCHMARK: Baseline vs. 2X Augmented vs. ONC Parity Matched")
    print("=" * 85)

    print("\n--- A. Training Baseline Models (N = 1,934: 967 MC vs 967 ONC) ---")
    results_base, _ = train_evaluate_morphometrics_pipeline(
        df=cohort_baseline,
        classes_to_include=["mc", "onc"],
        target_class="mc",
        cv_folds=5,
        run_loocv=False,
        random_state=42
    )
    results_base["Cohort"] = "Baseline (N=1934)"

    print("\n--- B. Training 2X Augmented Models (N = 3,868: 1,934 MC vs 1,934 ONC) ---")
    results_2x, _ = train_evaluate_morphometrics_pipeline(
        df=cohort_2x_balanced,
        classes_to_include=["mc", "onc"],
        target_class="mc",
        cv_folds=5,
        run_loocv=False,
        random_state=42
    )
    results_2x["Cohort"] = "2X Augmented (N=3868)"

    print("\n--- C. Training ONC Parity Matched Models (N = 7,280: 3,640 MC vs 3,640 ONC) ---")
    results_parity, artifacts_prod = train_evaluate_morphometrics_pipeline(
        df=cohort_parity_matched,
        classes_to_include=["mc", "onc"],
        target_class="mc",
        cv_folds=5,
        run_loocv=False,
        random_state=42
    )
    results_parity["Cohort"] = "ONC Parity Matched (N=7280)"

    comparison_df = pd.concat([results_base, results_2x, results_parity], ignore_index=True)
    comparison_csv = os.path.join(script_dir, "tps_mc_model_benchmark_comparison.csv")
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\nSaved benchmark comparison table to: {comparison_csv}")

    # 6. Diagnostic Visualizations
    print("\n[7] Generating Publication-Grade Diagnostic Visualizations...")

    # Figure 1: Wireframe Comparison (Real MC vs TPS Augmented MC)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    real_shapes, _ = extract_ordered_landmarks_20(mc_real_df)
    synth_shapes, _ = extract_ordered_landmarks_20(synth_mc_967_df)

    ax1, ax2 = axes
    plot_authentic_arm_wireframe(ax1, real_shapes[:40], color="#c0392b", alpha=0.35, lw=1.3, label="Authentic MC (Real, N=40 shown)")
    ax1.set_title("Authentic Malnourished Condition (MC) Wireframes\n(Field Clinical Cohort, N=967)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("X (Image Space Coordinate)", fontsize=11)
    ax1.set_ylabel("Y (Image Space Coordinate)", fontsize=11)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.4)

    plot_authentic_arm_wireframe(ax2, synth_shapes[:40], color="#8e44ad", alpha=0.35, lw=1.3, label="TPS-Augmented MC (Synthetic, N=40 shown)")
    ax2.set_title("Thin Plate Spline (TPS) Augmented MC Wireframes\n(Learned Non-Rigid Variations, N=967 Generated)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("X (Image Space Coordinate)", fontsize=11)
    ax2.set_ylabel("Y (Image Space Coordinate)", fontsize=11)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "fig1_authentic_arm_wireframe_comparison_mc.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"  -> Saved: {fig1_path}")

    # Figure 2: TPS Deformation Grid and Quiver Field
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ref_shape = augmentor.ssm.consensus_shape
    sample_target = augmentor.ssm.sample_shape(augmentor.rng, sigma_scale=1.5, max_std=2.5)

    tps_demo = ThinPlateSpline(alpha=0.005, dim=2)
    tps_demo.fit(ref_shape, sample_target)

    X_grid, Y_grid, X_warp, Y_warp = create_tps_deformation_grid(tps_demo, grid_size=20, margin_ratio=0.2)

    ax1 = axes[0]
    for row in range(X_grid.shape[0]):
        ax1.plot(X_warp[row, :], Y_warp[row, :], color="#2980b9", lw=0.9, alpha=0.7)
    for col in range(X_grid.shape[1]):
        ax1.plot(X_warp[:, col], Y_warp[:, col], color="#2980b9", lw=0.9, alpha=0.7)
    ax1.scatter(ref_shape[:, 0], ref_shape[:, 1], color="#27ae60", s=60, zorder=5, label="Source Consensus Landmarks")
    ax1.scatter(sample_target[:, 0], sample_target[:, 1], color="#e74c3c", s=60, marker="x", zorder=6, label="TPS Deformed MC Target")
    ax1.set_title(f"Thin Plate Spline 2D Continuous Deformation Grid\n(Bending Energy $E_b$ = {tps_demo.bending_energy:.5f}, $\\alpha=0.005$)", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.3)

    ax2 = axes[1]
    dx = sample_target[:, 0] - ref_shape[:, 0]
    dy = sample_target[:, 1] - ref_shape[:, 1]
    ax2.scatter(ref_shape[:, 0], ref_shape[:, 1], color="#27ae60", s=50, label="Reference Consensus")
    ax2.quiver(ref_shape[:, 0], ref_shape[:, 1], dx, dy, angles="xy", scale_units="xy", scale=1, color="#c0392b", width=0.007, label="Non-Rigid Displacement Field")
    ax2.scatter(sample_target[:, 0], sample_target[:, 1], color="#e74c3c", s=50, marker="x", label="Deformed MC Landmarks")
    ax2.set_title("Non-Rigid Landmark Displacement Vectors\n(Centripetal Soft-Tissue Wasting Inward Vectors)", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle=":", alpha=0.3)

    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "fig2_tps_deformation_grid_and_vectors_mc.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"  -> Saved: {fig2_path}")

    # Figure 3: PCA Morphospace Projection
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    onc_sample_plot = onc_real_df.sample(n=min(500, len(onc_real_df)), random_state=42)

    ax1 = axes[0]
    ax1.scatter(onc_sample_plot["PC1"], onc_sample_plot["PC2"], color="#27ae60", alpha=0.35, s=30, label="ONC (Well-Nourished)")
    ax1.scatter(mc_real_df["PC1"], mc_real_df["PC2"], color="#c0392b", alpha=0.6, s=35, label=f"Authentic MC (N={n_mc_real})")
    ax1.scatter(synth_mc_967_df["PC1"], synth_mc_967_df["PC2"], color="#8e44ad", alpha=0.5, s=25, marker="^", label="TPS Synthetic MC (N=967)")
    ax1.set_title("Procrustes PCA Morphospace: PC1 vs. PC2", fontsize=12, fontweight="bold")
    ax1.set_xlabel("PC1 (Anatomical Size / Posture Variation)", fontsize=10)
    ax1.set_ylabel("PC2 (Subcutaneous Tissue Thickness)", fontsize=10)
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(onc_sample_plot["PC2"], onc_sample_plot["PC3"], color="#27ae60", alpha=0.35, s=30, label="ONC (Well-Nourished)")
    ax2.scatter(mc_real_df["PC2"], mc_real_df["PC3"], color="#c0392b", alpha=0.6, s=35, label=f"Authentic MC (N={n_mc_real})")
    ax2.scatter(synth_mc_967_df["PC2"], synth_mc_967_df["PC3"], color="#8e44ad", alpha=0.5, s=25, marker="^", label="TPS Synthetic MC (N=967)")
    ax2.set_title("Procrustes PCA Morphospace: PC2 vs. PC3", fontsize=12, fontweight="bold")
    ax2.set_xlabel("PC2 (Subcutaneous Tissue Thickness)", fontsize=10)
    ax2.set_ylabel("PC3 (Arm Curvature / Forearm Angle)", fontsize=10)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "fig3_pca_morphospace_real_vs_synthetic_mc.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"  -> Saved: {fig3_path}")

    # Figure 4: Model Benchmark Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(13, 6))
    common_models = ["LDA", "Logistic Regression", "kNN (k=5)", "Random Forest", "XGBoost"]

    base_acc = [results_base[results_base["Classifier"] == m]["Accuracy"].values[0] * 100 for m in common_models]
    aug2x_acc = [results_2x[results_2x["Classifier"] == m]["Accuracy"].values[0] * 100 for m in common_models]
    parity_acc = [results_parity[results_parity["Classifier"] == m]["Accuracy"].values[0] * 100 for m in common_models]

    sens_key = [c for c in results_base.columns if "Sensitivity" in c][0]
    base_sens = [results_base[results_base["Classifier"] == m][sens_key].values[0] * 100 for m in common_models]
    aug2x_sens = [results_2x[results_2x["Classifier"] == m][sens_key].values[0] * 100 for m in common_models]
    parity_sens = [results_parity[results_parity["Classifier"] == m][sens_key].values[0] * 100 for m in common_models]

    x = np.arange(len(common_models))
    width = 0.13

    ax.bar(x - 2.5 * width, base_acc, width, label="Baseline Accuracy (N=1934)", color="#74b9ff")
    ax.bar(x - 1.5 * width, aug2x_acc, width, label="2X Aug. Accuracy (N=3868)", color="#0984e3")
    ax.bar(x - 0.5 * width, parity_acc, width, label="Parity Matched Accuracy (N=7280)", color="#2d3436")

    ax.bar(x + 0.5 * width, base_sens, width, label="Baseline MC Sensitivity", color="#fab1a0")
    ax.bar(x + 1.5 * width, aug2x_sens, width, label="2X Aug. MC Sensitivity", color="#e17055")
    ax.bar(x + 2.5 * width, parity_sens, width, label="Parity Matched MC Sensitivity", color="#d63031")

    ax.set_ylabel("Cross-Validated Metric Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Comparative Classification Gains: Baseline vs. 2X Augmented vs. ONC Parity Matched (5-Fold CV)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(common_models, fontsize=11, fontweight="bold")
    ax.set_ylim(60, 100)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "fig4_model_benchmark_comparison_bar_mc.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"  -> Saved: {fig4_path}")

    # Summary JSON
    summary_data = {
        "dataset_name": csv_name,
        "n_mc_authentic": n_mc_real,
        "n_onc_authentic": n_onc_real,
        "stage_1_2x": {
            "synthetic_samples": len(synth_mc_967_df),
            "total_mc_samples": len(mc_total_1934_df),
            "total_cohort_samples": len(cohort_2x_balanced)
        },
        "stage_2_onc_parity": {
            "synthetic_samples": len(synth_mc_2673_df),
            "total_mc_samples": len(mc_total_3640_df),
            "total_cohort_samples": len(cohort_parity_matched)
        },
        "ssm_mode_1_variance": float(augmentor.ssm.explained_variance_ratio[0]),
        "ssm_top_5_cumulative_variance": float(cum_var[4]),
        "ssm_top_10_cumulative_variance": float(cum_var[min(9, len(cum_var)-1)]),
        "centroid_size_mean": float(augmentor.cs_mean),
        "centroid_size_std": float(augmentor.cs_std),
        "muac_residual_std": float(augmentor.muac_residual_std)
    }
    summary_json_path = os.path.join(script_dir, "tps_mc_augmentation_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"Saved summary JSON to: {summary_json_path}")

    print("\n" + "=" * 85)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == "__main__":
    run_mc_tps_augmentation()
