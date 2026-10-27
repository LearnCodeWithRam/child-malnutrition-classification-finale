# Data-Driven Augmentation Parameter Estimation & Synthetic Data Generation
## Empirical Parameter Estimation & Generation for Landmark-Based Child Malnutrition Classification (ONC, MAM, SAM)

---

### 📌 Overview & Data Quality Filtering

In geometric morphometrics and out-of-sample landmark alignment benchmark studies, geometric data augmentation is frequently simulated using **arbitrary uniform distributions**:
- **Random Scale ($\beta$):** $\beta \sim U(0, 10)$
- **Random Rotation ($\theta$):** $\theta \sim U(0, 2\pi)$ ($-180^\circ$ to $+180^\circ$)
- **Random Translation ($\gamma$):** $\gamma \sim U(-5, 5)$
- **Gaussian Noise ($\sigma$):** Arbitrary Matrix Normal noise

However, raw annotation datasets can contain **vertically inverted image annotations** (where upper contour landmarks $Y_{12 \dots 20} < Y_{4 \dots 19}$ lower contour landmarks). Estimating parameters from distorted/inverted samples pollutes empirical parameter bounds and distorts Procrustes shape averages.

#### Automated Data Quality Filtering Applied:
- **Total Raw Samples:** 6,203 landmark configurations across child subjects (`final_cleaned_dataset_full.csv`).
- **Distorted / Vertically Inverted Samples Filtered Out:** **0 samples** (clean pre-filtered dataset).
- **Clean Quality Dataset Retained:** **6,203 samples** (ONC: 5,279, MAM Clean: 707, SAM Clean: 217).

Empirical augmentation parameters ($\beta, \theta, \gamma, \sigma$) are estimated strictly from the **6,203 clean configurations**.

---

### ⚙️ Global Generator & Class-Specific Empirical Parameters

A global synthetic data augmentation function `generate_class_specific_augmented_landmarks(class_type='sam', num_samples=500)` dynamically passes class-specific empirical parameter bounds depending on the requested class:

- **When generating SAM (`class_type='sam'`):** Uses SAM class-specific empirical parameters:
  - Scale: $\beta \sim U(0.70, 1.22)$
  - Rotation: $\theta \sim U(-7.0^\circ, +6.4^\circ)$
  - Translation: $\gamma_x \in [-5.05, +3.85], \gamma_y \in [-1.83, +2.39]$
  - Noise: $\sigma = 0.507$
- **When generating MAM (`class_type='mam'`):** Uses MAM class-specific empirical parameters:
  - Scale: $\beta \sim U(0.72, 1.22)$
  - Rotation: $\theta \sim U(-7.0^\circ, +6.4^\circ)$
  - Translation: $\gamma_x \in [-4.85, +3.72], \gamma_y \in [-2.33, +1.99]$
  - Noise: $\sigma = 0.454$
- **When generating ONC (`class_type='onc'`):** Uses ONC class-specific empirical parameters:
  - Scale: $\beta \sim U(0.70, 1.22)$
  - Rotation: $\theta \sim U(-7.0^\circ, +6.4^\circ)$
  - Translation: $\gamma_x \in [-4.85, +3.82], \gamma_y \in [-2.09, +2.14]$
  - Noise: $\sigma = 0.480$

---

### 📂 Generated Synthetic CSV Datasets

The pipeline generates and exports 4 balanced synthetic landmark CSV files matching the reference schema (`File_Name`, `Individual`, `Class_Type`, `X1`, `Y1`, ..., `X20`, `Y20`):

1. **`synthetic_augmented_onc_landmarks.csv`**: 500 generated synthetic ONC landmark samples.
2. **`synthetic_augmented_mam_landmarks.csv`**: 500 generated synthetic MAM landmark samples.
3. **`synthetic_augmented_sam_landmarks.csv`**: 500 generated synthetic SAM landmark samples.
4. **`synthetic_augmented_balanced_dataset_combined.csv`**: 1,500 total balanced synthetic landmark samples (ONC: 500, MAM: 500, SAM: 500).

---

### 🎨 Visualizations & Wireframe Diagrams

The pipeline produces 6 high-resolution, publication-quality figures:

1. **`fig1_wireframe_skeleton.png`**: Cartesian Landmark Plot with exact upper/lower edges and pairwise vertical cross-section lines.
2. **`fig2_class_consensus_shapes.png`**: 4-Panel Class-Wise Point Cloud Scatter & Consensus Wireframe Overlay ((a) ONC scatter $n=5279$, (b) MAM Clean scatter $n=707$, (c) SAM Clean scatter $n=217$, (d) Combined overlay).
3. **`fig3_landmark_variability_ellipses.png`**: Landmark Placement & Shape Dispersion 95% Confidence Ellipses across ONC, MAM, and SAM.
4. **`fig4_synthetic_augmentation_comparison.png`**: Visual Comparison between Generic Simulation Wireframe Cloud vs Empirical Data-Driven Cloud.
5. **`fig5_empirical_distributions.png`**: Empirical Parameter Distribution Boxplots & KDE Density curves by class.
6. **`fig6_generated_synthetic_wireframes.png`**: 4-Panel Generated Synthetic Wireframe Analysis ((a) Synthetic ONC wireframes, (b) Synthetic MAM wireframes, (c) Synthetic SAM wireframes, (d) Combined synthetic wireframes overlay).

---

### 📁 Directory Structure & File Manifest

```
Data-driven-augmentation(Estimate-para-from-real-dataset-itself)/version-2/
│
├── README.md                                                 # Comprehensive project documentation
├── Data_Driven_Augmentation_Parameter_Estimation_version_2.ipynb # Complete executable Jupyter Notebook
├── generate_plots.py                                         # Script performing analysis, generating figures & CSVs
├── empirical_augmentation_parameters_summary.json            # Machine-readable JSON summary of parameters
│
├── final_cleaned_dataset_full.csv                            # Clean reference landmark CSV dataset (N=6203)
│
├── synthetic_augmented_onc_landmarks.csv                     # Exported 500 synthetic ONC landmark samples
├── synthetic_augmented_mam_landmarks.csv                     # Exported 500 synthetic MAM landmark samples
├── synthetic_augmented_sam_landmarks.csv                     # Exported 500 synthetic SAM landmark samples
├── synthetic_augmented_balanced_dataset_combined.csv         # Combined 1500 balanced synthetic samples
│
├── fig1_wireframe_skeleton.png                               # Cartesian landmark plot with pairwise wireframe
├── fig2_class_consensus_shapes.png                           # 4-panel class-wise scatter & wireframe overlay
├── fig3_landmark_variability_ellipses.png                    # 95% confidence variability ellipses
├── fig4_synthetic_augmentation_comparison.png                # Generic vs Data-Driven synthetic wireframe clouds
├── fig5_empirical_distributions.png                          # Empirical distribution boxplots & KDE density plots
└── fig6_generated_synthetic_wireframes.png                  # 4-panel generated synthetic wireframe analysis
```

---

### 🚀 How to Run & Reproduce

#### 1. Execute Complete Parameter Estimation, Synthetic Generation & CSV Export
```bash
python generate_plots.py
```

#### 2. Open Interactive Notebook
Launch Jupyter Notebook or VS Code to run:
`Data_Driven_Augmentation_Parameter_Estimation_version_2.ipynb`

