# Geometric Transformation Data Augmentation: Train/Test Split & Class Balancing

## Executive Summary & Overview

This directory contains the pipeline, processed datasets, notebook, and evaluation reports for the **Global Geometric Transformation Modeling Data Augmentation Engine**, specifically configured for out-of-sample malnutrition classification across **ONC** (Other Normal Controls), **MAM** (Moderate Acute Malnutrition), and **SAM** (Severe Acute Malnutrition).

To avoid data leakage between training and testing folds:
1. **Master Clean Dataset (6,203 original specimens)** is partitioned into an **80% Training Set (4,963 specimens)** and a **20% Test Set (1,240 specimens)** using multi-variable stratified sampling across `Class_Type`, `age_group` ($\le 24$ vs $> 24$ months), and `gender` (`Female` vs `Male`).
2. **Test Set (1,240 specimens)** is kept strictly untouched as raw original baseline data.
3. **Training Set (4,963 specimens)** is augmented for minority classes (**MAM** and **SAM**) using class-specific empirical transformation parameters to achieve **707 samples per class**.

---

## 1. Stratified 80/20 Train-Test Split Breakdown

The 6,203 original clean specimens were split as follows:

| Clinical Class | Master Originals | 80% Train Set (Originals) | 20% Test Set (Untouched) |
| :--- | :---: | :---: | :---: |
| **ONC** | 5,279 | **4,223** | **1,056** |
| **MAM** | 707 | **566** | **141** |
| **SAM** | 217 | **174** | **43** |
| **Total** | **6,203** | **4,963** | **1,240** |

### Stratified Subgroup Breakdown

#### A. Train Set Subgroup Breakdown (Age x Gender)
Stratification maintains proportional distribution of age ($\le 24$ mo vs $> 24$ mo) and gender (`Female` vs `Male`) across all classes. Synthetic samples inherit their parent specimen's metadata:

| Class | Total Samples | $\le 24$ mo Female | $\le 24$ mo Male | $> 24$ mo Female | $> 24$ mo Male |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ONC (Original Baseline)** | 4,223 | 554 | 482 | 1,625 | 1,562 |
| **MAM (Originals)** | 566 | 30 | 76 | 232 | 228 |
| **MAM (Augmented Total)** | 707 | 40 | 98 | 282 | 287 |
| **SAM (Originals)** | 174 | 8 | 38 | 38 | 90 |
| **SAM (Augmented Total)** | 707 | 29 | 166 | 155 | 357 |

#### B. Test Set Subgroup Breakdown (Untouched Baseline)

| Class | Total Samples | $\le 24$ mo Female | $\le 24$ mo Male | $> 24$ mo Female | $> 24$ mo Male |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ONC** | 1,056 | 139 | 121 | 406 | 390 |
| **MAM** | 141 | 7 | 19 | 58 | 57 |
| **SAM** | 43 | 2 | 10 | 9 | 22 |

---

## 2. Class-Specific Empirical Transformation Parameters

The `generate_synthetic_geometric_augmentation` engine applies affine transformations ($\beta$ scaling, $\theta$ 2D rotation, $[\gamma_x, \gamma_y]$ translation, and $\sigma$ Gaussian digitization noise) based on empirical parameters calculated specifically for each clinical class:

### SAM Empirical Parameters (`class_type='sam'`)
- **Scale ($\beta$)**: $\beta \sim U(0.70, 1.22)$
- **Rotation ($\theta$)**: $\theta \sim U(-7.0^\circ, +6.4^\circ)$
- **Translation ($\gamma_x, \gamma_y$)**: $\gamma_x \in [-5.05, +3.85]$, $\gamma_y \in [-1.83, +2.39]$
- **Digitization Noise ($\sigma$)**: $\sigma = 0.507$ mm

### MAM Empirical Parameters (`class_type='mam'`)
- **Scale ($\beta$)**: $\beta \sim U(0.72, 1.22)$
- **Rotation ($\theta$)**: $\theta \sim U(-7.0^\circ, +6.4^\circ)$
- **Translation ($\gamma_x, \gamma_y$)**: $\gamma_x \in [-4.85, +3.72]$, $\gamma_y \in [-2.33, +1.99]$
- **Digitization Noise ($\sigma$)**: $\sigma = 0.454$ mm

### ONC Empirical Parameters (`class_type='onc'`)
- **Scale ($\beta$)**: $\beta \sim U(0.70, 1.22)$
- **Rotation ($\theta$)**: $\theta \sim U(-7.0^\circ, +6.4^\circ)$
- **Translation ($\gamma_x, \gamma_y$)**: $\gamma_x \in [-4.85, +3.82]$, $\gamma_y \in [-2.09, +2.14]$
- **Digitization Noise ($\sigma$)**: $\sigma = 0.480$ mm

---

## 3. Statistical Distribution Validation (Kolmogorov-Smirnov Test)

To ensure that synthetically generated shapes align with real clinical morphospace distributions, two-sample Kolmogorov-Smirnov (KS) tests were performed on Principal Component 1 (PC1):

- **SAM PC1 (Original Train vs Synthetic Train)**:
  - $D\text{-statistic} = 0.0526$
  - $p\text{-value} = 0.9583$ ($p > 0.05$, confirming statistical distribution alignment)
- **MAM PC1 (Original Train vs Synthetic Train)**:
  - $D\text{-statistic} = 0.0610$
  - $p\text{-value} = 0.8882$ ($p > 0.05$, confirming statistical distribution alignment)

---

## 4. File Registry

The following files are generated and stored in this folder:

| File Name | Description / Specimen Count |
| :--- | :--- |
| [`train_dataset_augmented.csv`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/Data-Augmentation-by-Geometric%20Transformation%20Modeling/train-test-split-and%20-augmentation/train_dataset_augmented.csv) | Main training dataset ($n=5,637$: 4,223 ONC, 707 MAM, 707 SAM) |
| [`train_dataset_1to1to1_balanced.csv`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/Data-Augmentation-by-Geometric%20Transformation%20Modeling/train-test-split-and%20-augmentation/train_dataset_1to1to1_balanced.csv) | Equal representation 1:1:1 training dataset ($n=2,121$: 707 ONC, 707 MAM, 707 SAM) |
| [`test_dataset_original.csv`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/Data-Augmentation-by-Geometric%20Transformation%20Modeling/train-test-split-and%20-augmentation/test_dataset_original.csv) | Untouched original test dataset ($n=1,240$: 1,056 ONC, 141 MAM, 43 SAM) |
| [`sam_augmented_train_geometric_samples.csv`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/Data-Augmentation-by-Geometric%20Transformation%20Modeling/train-test-split-and%20-augmentation/sam_augmented_train_geometric_samples.csv) | 533 synthetic SAM samples generated for training |
| [`mam_augmented_train_geometric_samples.csv`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/Data-Augmentation-by-Geometric%20Transformation%20Modeling/train-test-split-and%20-augmentation/mam_augmented_train_geometric_samples.csv) | 141 synthetic MAM samples generated for training |
| [`geometric_augmentation_train_test_summary_report.csv`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/Data-Augmentation-by-Geometric%20Transformation%20Modeling/train-test-split-and%20-augmentation/geometric_augmentation_train_test_summary_report.csv) | Full audit report containing pipeline parameters and KS test results |
| [`Geometric_Transformation_Data_Augmentation_train_test.ipynb`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/Data-Augmentation-by-Geometric%20Transformation%20Modeling/train-test-split-and%20-augmentation/Geometric_Transformation_Data_Augmentation_train_test.ipynb) | Jupyter Notebook executing the full split, augmentation, visualization, and export workflow |
| [`build_train_test_augmentation.py`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/Data-Augmentation-by-Geometric%20Transformation%20Modeling/train-test-split-and%20-augmentation/build_train_test_augmentation.py) | Python automation script to execute pipeline and generate notebook |
