# 📐 Geometric Morphometric Data Augmentation & Imbalance Classification Study

> **Directory**: `GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/geometric-dataAugmentation`  
> **Dataset Size**: $N = 6,335$ subjects  
> **Class Distribution**: Severe Acute Malnutrition (**SAM** = 220, 3.47%), Moderate Acute Malnutrition (**MAM** = 722, 11.40%), Optimal Nutrition Status (**ONC** = 5,393, 85.13%)  
> **Classifier**: XGBoost (`objective='multi:softprob'`, `eval_metric='mlogloss'`) evaluated on **Stratified 5-Fold Cross-Validation**  

---

## 📌 Executive Summary

Child malnutrition classification using 2D geometric morphometric shape coordinates presents an **extreme class imbalance challenge**. Evaluating models on raw overall accuracy is invalid because a trivial classifier predicting 100% ONC achieves **85.1% accuracy** while missing **100% of severe malnutrition (SAM) cases**.

Therefore, every experiment in this study is strictly evaluated on:
* **SAM Recall** (Sensitivity for severe malnutrition)
* **SAM Precision**
* **SAM F1-Score**
* **Macro F1-Score**
* **Balanced Accuracy**
* **Aggregated Confusion Matrix**

---

## 🧬 Feature Engineering & Noise Reduction

Following anatomical landmark quality inspection and collinearity analysis, landmarks **LM11**, **LM21**, and **LM22** (`GPA_X11`, `GPA_Y11`, `GPA_X21`, `GPA_Y21`, `GPA_X22`, `GPA_Y22`) were removed to eliminate digitizing noise and spatial redundancy.

* **Final Feature Vector (41 features)**:
  * Procrustes Shape Coordinates: `GPA_X1`..`GPA_Y20` (excluding LM11, LM21, LM22)
  * Size Feature: `Centroid_Size`
  * Metadata Features: `age_month`, `gender_num` (Male=0, Female=1)

---

## 🔄 Methodological Pipeline & Data Leakage Prevention

```mermaid
flowchart TD
    A["Raw Landmarks (20 points, 40 coords)"] --> B["Generalized Procrustes Analysis (GPA)"]
    B --> C["Remove Noisy Landmarks (LM11, LM21, LM22)"]
    C --> D["41 Procrustes & Metadata Features"]
    D --> E["Stratified 5-Fold Cross-Validation Split"]
    
    subgraph TrainFold["Inside Training Fold Only (Zero Leakage)"]
        E --> F["Resampling Method (SMOTE / Borderline / Weights)"]
        F --> G["Fit XGBoost Classifier"]
    end
    
    subgraph TestFold["Unseen Test Fold"]
        G --> H["Evaluate on Real Unseen Samples"]
    end
    
    H --> I["Aggregate Evaluation Metrics (SAM Recall, SAM F1, Macro F1, Bal Acc, CM)"]
```

> [!CAUTION]
> **Data Leakage Rule**: Resampling (SMOTE, BorderlineSMOTE, SMOTE-Tomek) is **never applied prior to train/test splitting**. Synthetic samples are generated strictly inside each training fold during cross-validation.

---

## 📊 Master Experiments Performance Comparison Table

| Experiment | Dataset Size | Target Data Ratio (ONC-MAM-SAM) | Augmentation Strategy | Synthetic % | SAM Recall | SAM Precision | SAM F1-Score | Macro F1 | Balanced Acc |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1: Baseline** | 6,335 | $5393 - 722 - 220$ | Class-Weighted XGBoost | 0.0% | 0.2409 | 0.6611 | 0.3527 | 0.6081 | 0.5584 |
| **Exp 2: Real Only (220)** 🥇 | 660 | $220 - 220 - 220$ | Pure Real Data Subsample | 0.0% | **0.6773** | 0.6328 | **0.6530** | 0.6415 | 0.6424 |
| **Exp 3: Limited Synthetic (500)** | 1,500 | $500 - 500 - 500$ | BorderlineSMOTE on SAM | 18.7% | 0.5682 | 0.5494 | 0.5581 | 0.6586 | 0.6601 |
| **Exp 4: Preferred Balanced (722)** 🥈 | 2,166 | $722 - 722 - 722$ | BorderlineSMOTE on SAM | 23.2% | 0.4909 | **0.6263** | 0.5473 | **0.6837** | **0.6733** |
| **Exp 5: SMOTE-Tomek (Full Data)** 🥉 | 6,335 | $5393 - 722 - 220$ | Dynamic SMOTE-Tomek | Dynamic | 0.4682 | 0.5253 | 0.4948 | 0.6705 | 0.6618 |
| **Exp 6: Borderline-SMOTE (Full Data)** | 6,335 | $5393 - 722 - 220$ | Dynamic Borderline-SMOTE | Dynamic | 0.4045 | 0.5869 | 0.4771 | 0.6621 | 0.6397 |

---

## 📓 Detailed Breakdown of Each Experiment

### Experiment 1: Original Data (Baseline - Class Weighted)
* **File**: [`Exp1_Baseline_Original_Data.ipynb`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/geometric-dataAugmentation/Exp1_Baseline_Original_Data.ipynb)
* **Dataset**: Full imbalanced dataset ($5,393$ ONC, $722$ MAM, $220$ SAM).
* **Augmentation**: None. Class weights computed using `compute_sample_weight('balanced', y_train)`.
* **Results**: SAM Recall = **0.2409** (24.09%), SAM Precision = **0.6611**, SAM F1 = **0.3527**, Macro F1 = **0.6081**, Balanced Acc = **0.5584**.
* **Key Finding**: Class-weighting alone fails on severe imbalance because majority class gradients dominate the tree splits, missing over 75% of SAM cases.

### Experiment 2: Pure Real-Data Balanced Subset (220-220-220) 🥇
* **File**: [`Exp2_Real_Data_Balanced_220.ipynb`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/geometric-dataAugmentation/Exp2_Real_Data_Balanced_220.ipynb)
* **Dataset**: Subsample of 220 ONC real, 220 MAM real, and all 220 SAM real ($N = 660$).
* **Augmentation**: 0% synthetic (100% biological data).
* **Results**: SAM Recall = **0.6773** (67.73%), SAM Precision = **0.6328**, SAM F1 = **0.6530**, Macro F1 = **0.6415**, Balanced Acc = **0.6424**.
* **Key Finding**: **Highest SAM Recall & SAM F1**. Eliminating majority class dominance allows the model to learn pure, uncorrupted biological shape differences between SAM, MAM, and ONC.

### Experiment 3: Limited Synthetic Expansion (500-500-500)
* **File**: [`Exp3_Limited_Synthetic_500.ipynb`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/geometric-dataAugmentation/Exp3_Limited_Synthetic_500.ipynb)
* **Dataset**: 500 ONC real, 500 MAM real, 220 SAM real + 280 BorderlineSMOTE SAM synthetic ($N = 1,500$).
* **Augmentation**: BorderlineSMOTE on SAM inside training folds up to 400 per fold.
* **Results**: SAM Recall = **0.5682** (56.82%), SAM Precision = **0.5494**, SAM F1 = **0.5581**, Macro F1 = **0.6586**, Balanced Acc = **0.6601**.
* **Key Finding**: Moderate synthetic expansion provides a good balance between recall (**56.82%**) and balanced accuracy.

### Experiment 4: Preferred Balanced Subset (722-722-722) 🥈
* **File**: [`Exp4_Preferred_Balanced_722.ipynb`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/geometric-dataAugmentation/Exp4_Preferred_Balanced_722.ipynb)
* **Dataset**: 722 ONC real, 722 MAM real (100% of real MAM), 220 SAM real + 502 BorderlineSMOTE SAM synthetic ($N = 2,166$).
* **Augmentation**: BorderlineSMOTE on SAM inside training folds to match 722.
* **Results**: SAM Recall = **0.4909** (49.09%), SAM Precision = **0.6263**, SAM F1 = **0.5473**, Macro F1 = **0.6837**, Balanced Acc = **0.6733**.
* **Key Finding**: **Highest Macro F1 (0.6837) & Balanced Accuracy (67.33%)**. Most publishable machine learning setup because it preserves 100% of available real MAM and SAM specimens while keeping synthetic augmentation moderate ($23.2\%$).

### Experiment 5: Full Data with SMOTE-Tomek (Training Fold Only) 🥉
* **File**: [`Exp5_Full_Data_SMOTE_Tomek.ipynb`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/geometric-dataAugmentation/Exp5_Full_Data_SMOTE_Tomek.ipynb)
* **Dataset**: Full dataset ($5,393$ ONC, $722$ MAM, $220$ SAM).
* **Augmentation**: Dynamic SMOTE-Tomek oversampling + Tomek link noise removal inside training folds.
* **Results**: SAM Recall = **0.4682** (46.82%), SAM Precision = **0.5253**, SAM F1 = **0.4948**, Macro F1 = **0.6705**, Balanced Acc = **0.6618**.
* **Key Finding**: Top full-dataset model. Tomek link cleaning removes overlapping boundary noise between SAM and MAM, nearly doubling SAM Recall over baseline (**46.82% vs 24.09%**).

### Experiment 6: Full Data with Borderline-SMOTE (Training Fold Only)
* **File**: [`Exp6_Full_Data_Borderline_SMOTE.ipynb`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/geometric-dataAugmentation/Exp6_Full_Data_Borderline_SMOTE.ipynb)
* **Dataset**: Full dataset ($5,393$ ONC, $722$ MAM, $220$ SAM).
* **Augmentation**: Dynamic Borderline-SMOTE targeting samples near class decision boundaries inside training folds.
* **Results**: SAM Recall = **0.4045** (40.45%), SAM Precision = **0.5869**, SAM F1 = **0.4771**, Macro F1 = **0.6621**, Balanced Acc = **0.6397**.
* **Key Finding**: Achieves higher precision (**58.69%**) than standard SMOTE by focusing synthetic generation along boundary regions.

---

## 💡 Scientific Conclusions & Recommendations

1. **🏆 Highest SAM Sensitivity & Cleanest Benchmark: Experiment 2 (220-220-220 Real Data)**
   * **SAM Recall**: **67.73%** | **SAM F1**: **0.6530**
   * *Biological Justification*: By using equal 220 real specimens for each class, the model learns unbiased shape differences without synthetic noise. It detects over two-thirds of severe malnutrition cases.

2. **🥈 Most Publishable & Preferred Model: Experiment 4 (722-722-722 Preferred Balanced)**
   * **Macro F1**: **0.6837** | **Balanced Acc**: **0.6733** | **SAM Precision**: **62.63%**
   * *ML Justification*: Retains **100% of available real MAM and SAM specimens**, uses only a moderate amount of synthetic SAM data ($23.2\%$), and delivers high overall macro accuracy while maintaining strong SAM detection.

3. **🥉 Full Dataset Winner: Experiment 5 (Full Data + SMOTE-Tomek)**
   * **Macro F1**: **0.6705** | **Balanced Acc**: **0.6618** | **SAM Recall**: **46.82%**
   * *ML Justification*: Demonstrates that SMOTE-Tomek dynamic oversampling + Tomek link noise removal outperforms standard class-weighted XGBoost (Baseline SAM Recall was only 24.09%).

---

## 📁 File Manifest

* [`Exp1_Baseline_Original_Data.ipynb`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/geometric-dataAugmentation/Exp1_Baseline_Original_Data.ipynb)
* [`Exp2_Real_Data_Balanced_220.ipynb`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/geometric-dataAugmentation/Exp2_Real_Data_Balanced_220.ipynb)
* [`Exp3_Limited_Synthetic_500.ipynb`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/geometric-dataAugmentation/Exp3_Limited_Synthetic_500.ipynb)
* [`Exp4_Preferred_Balanced_722.ipynb`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/geometric-dataAugmentation/Exp4_Preferred_Balanced_722.ipynb)
* [`Exp5_Full_Data_SMOTE_Tomek.ipynb`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/geometric-dataAugmentation/Exp5_Full_Data_SMOTE_Tomek.ipynb)
* [`Exp6_Full_Data_Borderline_SMOTE.ipynb`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/geometric-dataAugmentation/Exp6_Full_Data_Borderline_SMOTE.ipynb)
* [`Master_Augmentation_Experiments_Comparison.ipynb`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/geometric-dataAugmentation/Master_Augmentation_Experiments_Comparison.ipynb)
* [`all_experiments_results.csv`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/geometric-dataAugmentation/all_experiments_results.csv)
* [`confusion_matrices.json`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/geometric-dataAugmentation/confusion_matrices.json)
* [`generate_all_notebooks.py`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/geometric-dataAugmentation/generate_all_notebooks.py)
* [`populate_all_notebook_outputs.py`](file:///e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/geometric-dataAugmentation/populate_all_notebook_outputs.py)
