# Geometric Morphometrics Thin Plate Spline (TPS) Data Augmentation & Universal Classification Pipeline
## Severe Acute Malnutrition (SAM) Left-Arm Shape Dataset Expansion to 600 Samples & Beyond

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-red.svg)](https://xgboost.readthedocs.io/)
[![Nature Scientific Reports](https://img.shields.io/badge/DOI-10.1038%2Fs41598--025--85718--4-darkblue.svg)](https://doi.org/10.1038/s41598-025-85718-4)
[![IEEE Access](https://img.shields.io/badge/Reference-Tang%20et%20al.%20(2019)-green.svg)](https://doi.org/10.1109/ACCESS.2019.2941147)
[![GitHub TPS](https://img.shields.io/badge/GitHub-raphaelreme%2Ftps-purple.svg)](https://github.com/raphaelreme/tps.git)

---

## 1. Executive Summary & Problem Context

In clinical mobile screening for child malnutrition (**SAM Photo Diagnosis App Program**, *Action Against Hunger* & *Universidad Carlos III de Madrid*), severe class imbalance is a fundamental barrier to training robust machine learning models:
* **Optimal Nutritional Condition (ONC)**: $N = 5,248$
* **Moderate Acute Malnutrition (MAM)**: $N = 741$
* **Severe Acute Malnutrition (SAM)**: $N = 214$ (Only ~3.4% of screening records)

Because child left-arm shape involves continuous, non-rigid anatomical deformations (tissue atrophy, muscular thinning, and postural bending), naive data augmentations (such as random Gaussian jitter, affine rotation, or flipping) produce non-biological shapes that violate anatomical constraints and break clinical validity.

This repository implements an authentic, publication-grade augmentation pipeline combining **Statistical Shape Modeling (SSM)** (*Tang et al., IEEE Access 2019*) with **Thin Plate Spline (TPS) Non-Rigid Warping** (*Bookstein 1989; Raphael Reme / `tps`*), complete 104-column feature recalculation, and a universal global API.

---

## 2. Directory Structure & Key Files

All resources are located in:
`GM_out_of_sample_classification-main\SAM-Photo\thin_plate_sphine_based_data_augmentation\`

```
thin_plate_sphine_based_data_augmentation/
├── README.md                                                        <- This comprehensive documentation file
├── TPS_Based_Data_Augmentation_and_Model_Training.ipynb             <- Interactive Jupyter Notebook with inline visualizations
├── run_tps_augmentation_sam600.py                                   <- Master executable end-to-end pipeline script
├── tps_math.py                                                      <- 2D/3D Thin Plate Spline math engine & bending energy solver
├── tps_augmentation.py                                              <- Statistical Shape Model & TPS Augmentation Engine
├── tps_model_training.py                                            <- Morphometrics ML training, fGPA, allometry & CV pipeline
│
├── final_cleaned_dataset_full_with_muac_corrected_with_who_R_package.csv <- Source clinical dataset (N = 6,203, 104 cols)
├── sam_augmented_600_total.csv                                      <- Exact 600 SAM dataset (214 real + 386 synthetic)
├── sam_pure_synthetic_600.csv                                       <- 600 pure synthetic SAM samples generated via SSM+TPS
├── final_cleaned_dataset_full_with_sam_augmented_600.csv             <- Full screening dataset with SAM augmented to 600 (N = 6,589)
├── final_cleaned_dataset_balanced_600_sam_onc.csv                   <- Balanced clinical benchmark cohort (600 SAM vs. 600 ONC)
├── tps_model_benchmark_comparison.csv                               <- Baseline vs. TPS-Augmented benchmark results
├── tps_augmentation_summary.json                                    <- Mathematical parameters, variance explained & metadata
│
└── figures/
    ├── fig1_authentic_arm_wireframe_comparison.png                  <- Original vs. TPS-Augmented SAM left-arm wireframes
    ├── fig2_tps_deformation_grid_and_vectors.png                    <- Continuous 2D TPS warping grid & displacement field
    ├── fig3_pca_morphospace_real_vs_synthetic.png                   <- Procrustes PCA shape space (PC1 vs PC2 & PC2 vs PC3)
    └── fig4_model_benchmark_comparison_bar.png                      <- Grouped comparative bar chart of ML performance gains
```

---

## 3. Scientific & Mathematical Foundations

### 3.1 Thin Plate Spline (TPS) Physical Formulation
Thin Plate Splines model the physical bending of a thin sheet of metal clamped at control points, minimizing the total bending energy:

$$E_b = \iint_{\mathbb{R}^2} \left( \left(\frac{\partial^2 f}{\partial x^2}\right)^2 + 2\left(\frac{\partial^2 f}{\partial x \partial y}\right)^2 + \left(\frac{\partial^2 f}{\partial y^2}\right)^2 \right) dx\,dy$$

For 2D landmark coordinates, the radial basis function kernel is:
$$U(r) = r^2 \ln(r) \quad (r > 0), \quad U(0) = 0$$

For 3D volumetric coordinates (*Tang et al., 2019*), the biharmonic kernel is:
$$U(r) = |r|$$

Given source control points $P = (p_1, \dots, p_k)^T \in \mathbb{R}^{k \times 2}$ and target points $Q = (q_1, \dots, q_k)^T \in \mathbb{R}^{k \times 2}$, the non-rigid mapping is obtained by solving the closed-form partitioned linear system:

$$\begin{bmatrix} K + \alpha I & P' \\ P'^T & \mathbf{0}_{3 \times 3} \end{bmatrix} \begin{bmatrix} W \\ A \end{bmatrix} = \begin{bmatrix} Q \\ \mathbf{0}_{3 \times 2} \end{bmatrix}$$

where $K_{ij} = U(\|p_i - p_j\|)$, $P' = [\mathbf{1}, P]$, $\alpha \ge 0$ is the regularization smoothness parameter, $W$ are non-affine warping weights, and $A$ are affine coefficients. The continuous deformation field is:

$$f(x) = [1, x_1, x_2] A + \sum_{i=1}^k w_i U(\|x - p_i\|)$$

### 3.2 Learning Biological Variation via Statistical Shape Modeling (SSM)
Following *Tang et al. (IEEE Access, 2019)*, we extract authentic anatomical variations from the class cohort:
1. **Full Generalized Procrustes Analysis (fGPA)**: Strips camera translation and scaling to align configurations to the Fréchet mean consensus shape $\bar{S}$.
2. **Covariance Decomposition**: Computes the shape space covariance matrix $\Sigma = \frac{1}{N-1} (Z - \bar{Z})^T (Z - \bar{Z})$ across the landmark coordinates.
3. **Principal Variation Modes**: Eigendecomposition $\Sigma = \Phi \Lambda \Phi^T$ yields eigenvalues $\lambda_k$ and orthogonal eigenvectors $\phi_k$.
   - **Mode 1 ($33.26\%$ of variance)**: Symmetric centripetal wasting along the upper arm contour ($L_{12}, L_{14}, L_{16}, L_{18}$ moving inward toward $L_{4}, L_{13}, L_{15}, L_{17}$).
   - **Top 5 Modes**: Account for **$76.12\%$** of cumulative anatomical variance.
   - **Top 10 Modes**: Account for **$89.47\%$** of cumulative variance.
4. **Physiologically Bounded Sampling**:
   $$Z_{\text{synth}} = \bar{Z} + \sum_{k=1}^m b_k \sqrt{\lambda_k} \phi_k, \quad b_k \sim \mathcal{N}(0, \sigma^2), \quad |b_k| \le 2.8$$

### 3.3 Coupled Anthropometric & Clinical Synthesis
Each synthetic specimen is coupled with clinically realistic derived features:
* **Centroid Size ($CS$)**: Scaled from the empirical class distribution: $CS \sim \mathcal{N}(\mu_{CS}=39.75, \sigma_{CS}=6.43)$.
* **Clinical MUAC ($cm$)**: Couplings mid-upper arm thickness ($\|p_{16} - p_{15}\|$) and $CS$ via multivariate regression with calibrated biological residual noise ($\sigma = 1.54\text{ cm}$).
* **Age & Biological Sex**: Stratified across the 5 WHO clinical brackets ($6-12m, 13-24m, 25-36m, 37-48m, 49-59m$) matching observed prevalence ($74.8\%$ Male, $25.2\%$ Female).

---

## 4. Generated Datasets Summary

| File Name | Description | Rows ($N$) | Columns | Status |
| :--- | :--- | :---: | :---: | :---: |
| [`sam_augmented_600_total.csv`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/SAM-Photo/thin_plate_sphine_based_data_augmentation/sam_augmented_600_total.csv) | Exact 600 SAM dataset (**214 authentic original + 386 TPS synthetic**) | **600** | 104 | Generated |
| [`sam_pure_synthetic_600.csv`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/SAM-Photo/thin_plate_sphine_based_data_augmentation/sam_pure_synthetic_600.csv) | 600 pure synthetic SAM samples generated via SSM+TPS | **600** | 104 | Generated |
| [`final_cleaned_dataset_full_with_sam_augmented_600.csv`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/SAM-Photo/thin_plate_sphine_based_data_augmentation/final_cleaned_dataset_full_with_sam_augmented_600.csv) | Full multi-class screening dataset with SAM augmented to 600 | **6,589** | 104 | Generated |
| [`final_cleaned_dataset_balanced_600_sam_onc.csv`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/SAM-Photo/thin_plate_sphine_based_data_augmentation/final_cleaned_dataset_balanced_600_sam_onc.csv) | Strictly balanced clinical benchmark cohort (**600 SAM vs. 600 ONC**) | **1,200** | 104 | Generated |

### Verified Column Schema (All 104 Fields Preserved):
1. **Metadata**: `File_Name`, `Individual`, `Class_Type`, `age_month`, `gender`, `muac_cm`.
2. **Quality & Error Checks**: `missing_count`, `zero_coord_count`, `Digitizing_Error_Flag`, `Procrustes_Outlier`, `PCA_Outlier`, `Mahalanobis_Outlier`.
3. **Anatomical Landmarks**: $X1, Y1, X2, Y2, \dots, X20, Y20$ (19 pairs present; $L_{11}$ reconstructed as forearm midpoint).
4. **Size Metric**: `Centroid_Size` ($CS = \sqrt{\sum (x_i - \bar{x})^2 + (y_i - \bar{y})^2}$).
5. **Generalized Procrustes Coordinates**: $GPA\_X1, GPA\_Y1, \dots, GPA\_X20, GPA\_Y20$.
6. **Procrustes Metrics & PCA Space**: `Procrustes_Distance`, `Mahalanobis_Distance`, `PC1` through `PC10`.
7. **Inter-Landmark Distances (ILD)**: `ILD_Shoulder_Wrist`, `ILD_Shoulder_Armpit`, `Bounding_Box_Area`.

---

## 5. Machine Learning Benchmark: Baseline vs. TPS-Augmented

Stratified 5-Fold Cross-Validation evaluation comparing **Baseline (Unaugmented, $N=428$)** vs. **TPS-Augmented ($N=1200$)** cohorts (*SAM vs. ONC*):

| Classifier | Baseline Acc ($N=428$) | Augmented Acc ($N=1200$) | Net Gain | Baseline SAM Recall (Sens) | Augmented SAM Recall (Sens) | Baseline ROC-AUC | Augmented ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Gradient Boosted)** | 71.39% | **83.97%** | **+12.58%** | 70.62% | **80.00%** | 0.796 | **0.910** |
| **k-Nearest Neighbors (k=5)** | 70.11% | **81.55%** | **+11.44%** | 76.82% | 76.03% | 0.766 | **0.890** |
| **Random Forest (100 trees)** | 72.69% | **78.53%** | **+5.84%** | 69.08% | **73.62%** | 0.805 | **0.859** |
| **LDA (Linear Discriminant)** | 64.94% | **68.88%** | **+3.94%** | 66.05% | **72.07%** | 0.695 | **0.762** |
| **Logistic Regression** | 69.57% | **69.66%** | +0.09% | 59.78% | **63.45%** | 0.766 | 0.747 |

### Clinical Benchmark Insights:
1. **XGBoost Accuracy jumps from 71.39% to 83.97% (+12.58%)**, and ROC-AUC reaches **0.910**.
2. **Clinical SAM Sensitivity (Recall)** jumps to **80.00%** on XGBoost and **72.07%** on LDA, critically reducing life-threatening false negatives.
3. **kNN Accuracy jumps by +11.44%** (from 70.11% to 81.55%), confirming that TPS generates dense, continuous coverage of the SAM shape manifold without creating isolated outliers.

---

## 6. Diagnostic Visualizations

All diagnostic plots are saved in [`figures/`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/SAM-Photo/thin_plate_sphine_based_data_augmentation/figures):

1. **[`fig1_authentic_arm_wireframe_comparison.png`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/SAM-Photo/thin_plate_sphine_based_data_augmentation/figures/fig1_authentic_arm_wireframe_comparison.png)**:
   - Panel A: Authentic SAM arm wireframes ($N=214$ cohort).
   - Panel B: TPS-augmented SAM wireframes ($N=386$ generated).
   - Proves anatomical integrity: upper arm contour, lower arm contour, shoulder boundary, and wrist closure are preserved.
2. **[`fig2_tps_deformation_grid_and_vectors.png`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/SAM-Photo/thin_plate_sphine_based_data_augmentation/figures/fig2_tps_deformation_grid_and_vectors.png)**:
   - Panel A: Continuous 2D TPS elastic grid deformation ($E_b = 0.00192$, $\alpha=0.005$).
   - Panel B: Quiver plot showing centripetal tissue retraction vectors characteristic of subcutaneous fat and muscle wasting.
3. **[`fig3_pca_morphospace_real_vs_synthetic.png`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/SAM-Photo/thin_plate_sphine_based_data_augmentation/figures/fig3_pca_morphospace_real_vs_synthetic.png)**:
   - Panel A: PC1 vs. PC2 shape space.
   - Panel B: PC2 vs. PC3 shape space.
   - Confirms synthetic SAM instances cluster within the authentic SAM biological subspace with zero outliers.
4. **[`fig4_model_benchmark_comparison_bar.png`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/SAM-Photo/thin_plate_sphine_based_data_augmentation/figures/fig4_model_benchmark_comparison_bar.png)**:
   - Grouped bar chart illustrating Accuracy and Sensitivity gains across LDA, LR, kNN, RF, and XGBoost.

---

## 7. Universal Global API (How to Use for ANY Class or Dataset)

The pipeline is completely generic and can be applied to **ANY** class or dataset:

```python
import pandas as pd
from tps_augmentation import augment_class_dataset, augment_multi_class_dataset
from tps_model_training import train_evaluate_morphometrics_pipeline

# 1. Load your dataset
df = pd.read_csv("final_cleaned_dataset_full_with_muac_corrected_with_who_R_package.csv")

# ------------------------------------------------------------------------------
# Method A: Augment ANY Specific Class (e.g. SAM to 600, or MAM to 1,000)
# ------------------------------------------------------------------------------
synth_df, combined_df = augment_class_dataset(
    df=df,
    target_class="sam",          # Any class name: 'sam', 'mam', 'onc', or custom
    target_total_samples=600,    # Target total sample count
    method="ssm_tps",            # 'ssm_tps' (Tang et al.), 'pairwise_tps', or 'elastic_tps'
    smoothness=0.005,            # TPS alpha regularization parameter
    random_state=42
)

# ------------------------------------------------------------------------------
# Method B: Multi-Class Augmentation (Balance multiple classes simultaneously)
# ------------------------------------------------------------------------------
balanced_df = augment_multi_class_dataset(
    df=df,
    target_counts={"sam": 600, "mam": 600, "onc": 600},
    method="ssm_tps"
)

# ------------------------------------------------------------------------------
# Method C: Universal Model Training & Cross-Validation
# ------------------------------------------------------------------------------
results_df, artifacts = train_evaluate_morphometrics_pipeline(
    df=balanced_df,
    class_column="Class_Type",
    classes_to_include=["sam", "onc"],  # or ["sam", "mam", "onc"] for 3-class
    target_class="sam",
    use_allometry=True,                 # Multivariate regression on log(CS)
    use_18_landmarks=True,              # Drops collinear semilandmarks 2 and 3
    include_muac=True,                  # Multimodal feature fusion (+ MUAC cm)
    cv_folds=5,
    run_loocv=False,
    random_state=42
)

print(results_df)
```

---

## 7.5 MC Class (SAM + MAM) 2X Augmentation & ONC Parity Matching

In addition to SAM 600 sample expansion, the universal pipeline supports direct augmentation of the merged **Malnourished Condition (MC = SAM + MAM, $N=967$)** to resolve class imbalance against the **Optimal Nutritional Condition (ONC, $N=3,640$)** cohort:

### Generated MC Datasets
| File Name | Description | Rows ($N$) | Columns | Status |
| :--- | :--- | :---: | :---: | :---: |
| [`mc_pure_synthetic_2x_967.csv`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/SAM-Photo/thin_plate_sphine_based_data_augmentation/mc_pure_synthetic_2x_967.csv) | Stage 1 pure synthetic MC specimens | **967** | 105 | Generated |
| [`mc_augmented_2x_total_1934.csv`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/SAM-Photo/thin_plate_sphine_based_data_augmentation/mc_augmented_2x_total_1934.csv) | Stage 1 2X MC dataset (**967 real + 967 synthetic**) | **1,934** | 105 | Generated |
| [`final_cleaned_dataset_augmented_2x_mc1934_onc1934.csv`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/SAM-Photo/thin_plate_sphine_based_data_augmentation/final_cleaned_dataset_augmented_2x_mc1934_onc1934.csv) | Stage 1 balanced 2X benchmark cohort (**1,934 MC vs. 1,934 ONC**) | **3,868** | 105 | Generated |
| [`mc_pure_synthetic_onc_match_2673.csv`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/SAM-Photo/thin_plate_sphine_based_data_augmentation/mc_pure_synthetic_onc_match_2673.csv) | Stage 2 pure synthetic MC specimens matching ONC count | **2,673** | 105 | Generated |
| [`mc_augmented_onc_match_total_3640.csv`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/SAM-Photo/thin_plate_sphine_based_data_augmentation/mc_augmented_onc_match_total_3640.csv) | Stage 2 ONC-matched MC cohort (**967 real + 2,673 synthetic**) | **3,640** | 105 | Generated |
| [`final_cleaned_dataset_augmented_parity_mc3640_onc3640.csv`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/SAM-Photo/thin_plate_sphine_based_data_augmentation/final_cleaned_dataset_augmented_parity_mc3640_onc3640.csv) | Full matched parity benchmark cohort (**3,640 MC vs. 3,640 ONC**) | **7,280** | 105 | Generated |

### Multi-Classifier Empirical Benchmark: Baseline vs. 2X Augmented vs. ONC Parity Matched (5-Fold CV)
| Classifier | Baseline Acc ($N=1934$) | 2X Aug. Acc ($N=3868$) | Parity Matched Acc ($N=7280$) | Baseline MC Recall | 2X MC Recall | Parity MC Recall | Baseline ROC-AUC | Parity ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | 79.68% | 86.62% | **91.36%** (+11.68%) | 75.68% | 81.44% | **85.73%** | 0.883 | **0.960** |
| **kNN (k=5)** | 78.53% | 86.12% | **90.83%** (+12.30%) | 78.97% | 83.24% | **86.49%** | 0.864 | **0.955** |
| **Random Forest** | 80.01% | 83.54% | **88.83%** (+8.82%) | 76.77% | 79.63% | **84.60%** | 0.881 | **0.953** |
| **LDA** | 77.98% | 78.91% | **79.79%** (+1.81%) | 78.42% | 79.41% | **80.85%** | 0.843 | **0.874** |
| **Logistic Regression**| 72.56% | 75.93% | **76.65%** (+4.09%) | 64.61% | 71.22% | **73.80%** | 0.833 | **0.834** |

---

## 8. How to Execute & Replicate

### Option 1: Execute MC Class 2X & Parity Augmentation from CLI
```bash
python run_tps_augmentation_mc_onc.py
```

### Option 2: Execute SAM Expansion to 600 from CLI
```bash
python run_tps_augmentation_sam600.py
```

### Option 3: Run Interactive Jupyter Notebook
```bash
jupyter notebook TPS_Based_Data_Augmentation_and_Model_Training.ipynb
```

---

## 9. Academic Citations & References

1. **Tang, Z., Chen, K., Pan, M., Wang, M., & Song, Z.** (2019). *"An Augmentation Strategy for Medical Image Processing Based on Statistical Shape Model and 3D Thin Plate Spline for Deep Learning"*. **IEEE Access**, 7, 133111–133121. [DOI: 10.1109/ACCESS.2019.2941147](https://doi.org/10.1109/ACCESS.2019.2941147).
2. **Medialdea, L., Arribas-Gil, A., Pérez-Romero, Á., & Gómez, A.** (2025). *"Geometric morphometrics approach for classifying children’s nutritional status on out of sample data"*. **Scientific Reports (Nature)**, 15:3906. [DOI: 10.1038/s41598-025-85718-4](https://doi.org/10.1038/s41598-025-85718-4).
3. **Bookstein, F. L.** (1989). *"Principal Warps: Thin-plate splines and the decomposition of deformations"*. **IEEE Transactions on Pattern Analysis and Machine Intelligence**, 11(6), 567–585.
4. **Dryden, I. L., & Mardia, K. V.** (2016). *"Statistical Shape Analysis: With Applications in R"*. John Wiley & Sons.
5. **Rème, R.** *Thin Plate Spline (TPS) / Generalized Polyharmonic Spline Interpolation*. GitHub: [raphaelreme/tps](https://github.com/raphaelreme/tps.git).
