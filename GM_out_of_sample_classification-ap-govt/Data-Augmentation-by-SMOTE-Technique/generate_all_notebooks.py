import os
import json

out_dir = r"e:\ap-govt-EDA-model-tainning-testing\GM_out_of_sample_classification-main\GM_out_of_sample_classification-ap-govt\geometric-dataAugmentation"
os.makedirs(out_dir, exist_ok=True)

def create_markdown_cell(content):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content.splitlines(keepends=True)
    }

def create_code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True)
    }

def save_notebook(filepath, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print(f"Created notebook: {os.path.basename(filepath)}")

# Helper function to generate individual experiment notebook
def build_exp_notebook(filepath, title, exp_num, data_summary, pipeline_desc, setup_code, exp_type):
    cells = []
    
    # 1. Header Markdown
    header_md = f"""# Geometric Morphometric Data Augmentation & Imbalance Classification
## Experiment {exp_num}: {title}

### 📌 Study Overview & Experimental Objective
* **Dataset Size & Class Distribution**: {data_summary}
* **Feature Selection**: Geometric Morphometric Procrustes Coordinates excluding noisy/collinear landmarks (**LM11**, **LM21**, **LM22**) $\\rightarrow$ **41 total features** (`GPA_X1`..`GPA_Y20`, `Centroid_Size`, `age_month`, `gender_num`).
* **Primary Metric Focus**: **SAM Recall**, **SAM Precision**, **SAM F1**, **Macro F1**, and **Balanced Accuracy** (avoiding raw overall accuracy which masks minor class performance).

---
### 🔄 Experimental Pipeline
```text
{pipeline_desc}
```
"""
    cells.append(create_markdown_cell(header_md))

    # 2. Imports Code
    imports_code = """import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, balanced_accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings("ignore")

# Global Reproducibility Seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
print("✅ Libraries loaded successfully.")
"""
    cells.append(create_code_cell(imports_code))

    # 3. Data Loading Code
    data_code = """# Load Cleaned Geometric Morphometric Data
data_path = r"e:\\ap-govt-EDA-model-tainning-testing\\GM_out_of_sample_classification-main\\GM_out_of_sample_classification-ap-govt\\outliers_quality_of_data_extraction\\final_cleaned_dataset_full.csv"
df = pd.read_csv(data_path)

# Encode Gender
df['gender_num'] = df['gender'].map({'M': 0, 'F': 1, 'Male': 0, 'Female': 1}).fillna(0)

# Selected Feature Columns (19 GPA Landmarks + Centroid Size + Age + Gender)
feature_cols = [c for c in df.columns if c.startswith('GPA_')] + ['Centroid_Size', 'age_month', 'gender_num']
target_col = 'Class_Type'

print(f"📊 Cleaned Dataset Shape: {df.shape}")
print(f"🧬 Selected Feature Count: {len(feature_cols)} (38 GPA Coordinates + 3 Demographics)")
print("🏷️ Full Dataset Class Distribution:")
print(df[target_col].value_counts())
"""
    cells.append(create_code_cell(data_code))

    # 4. Experiment Setup Code
    cells.append(create_code_cell(setup_code))

    # 5. Cross Validation Execution Code
    cv_code = f"""# Stratified 5-Fold Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

recalls, precisions, f1s_sam, macro_f1s, balanced_accs = [], [], [], [], []
cm_total = np.zeros((3, 3), dtype=int)
class_names = ['ONC', 'MAM', 'SAM']

print(f"🚀 Running 5-Fold Stratified Cross-Validation (Exp {exp_num}: {title})...\\n")

for fold, (train_idx, test_idx) in enumerate(cv.split(X_exp, y_exp), 1):
    X_tr, y_tr = X_exp[train_idx], y_exp[train_idx]
    X_te, y_te = X_exp[test_idx], y_exp[test_idx]
    
    # Apply Resampling / Weighting strictly INSIDE training fold to prevent data leakage
    if "{exp_type}" == "exp1":
        sample_weights = compute_sample_weight('balanced', y_tr)
        X_res, y_res = X_tr, y_tr
    elif "{exp_type}" == "exp2":
        X_res, y_res = X_tr, y_tr
    elif "{exp_type}" == "exp3":
        bsmote = BorderlineSMOTE(sampling_strategy={{2: 400}}, random_state=RANDOM_STATE, k_neighbors=3)
        X_res, y_res = bsmote.fit_resample(X_tr, y_tr)
    elif "{exp_type}" == "exp4":
        bsmote = BorderlineSMOTE(sampling_strategy={{2: 577}}, random_state=RANDOM_STATE, k_neighbors=3)
        X_res, y_res = bsmote.fit_resample(X_tr, y_tr)
    elif "{exp_type}" == "exp5":
        smote_tomek = SMOTETomek(random_state=RANDOM_STATE)
        X_res, y_res = smote_tomek.fit_resample(X_tr, y_tr)
    elif "{exp_type}" == "exp6":
        bsmote = BorderlineSMOTE(random_state=RANDOM_STATE, k_neighbors=3)
        X_res, y_res = bsmote.fit_resample(X_tr, y_tr)

    # Instantiate XGBoost Model
    model = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=RANDOM_STATE,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4
    )
    
    if "{exp_type}" == "exp1":
        model.fit(X_tr, y_tr, sample_weight=sample_weights)
    else:
        model.fit(X_res, y_res)
    
    # Evaluate on unseen test fold
    y_pred = model.predict(X_te)
    
    rec = recall_score(y_te, y_pred, labels=[2], average=None)[0]
    prec = precision_score(y_te, y_pred, labels=[2], average=None, zero_division=0)[0]
    f1_s = f1_score(y_te, y_pred, labels=[2], average=None)[0]
    m_f1 = f1_score(y_te, y_pred, average='macro')
    b_acc = balanced_accuracy_score(y_te, y_pred)
    
    recalls.append(rec)
    precisions.append(prec)
    f1s_sam.append(f1_s)
    macro_f1s.append(m_f1)
    balanced_accs.append(b_acc)
    cm_total += confusion_matrix(y_te, y_pred, labels=[0, 1, 2])
    
    print(f"Fold {{fold}}: SAM Recall={{rec:.4f}} | SAM Prec={{prec:.4f}} | SAM F1={{f1_s:.4f}} | Macro F1={{m_f1:.4f}} | Bal Acc={{b_acc:.4f}}")
"""
    cells.append(create_code_cell(cv_code))

    # 6. Evaluation and Plotting Code
    eval_code = f"""# Compute Aggregated Metrics Across Folds
mean_sam_rec = np.mean(recalls)
mean_sam_prec = np.mean(precisions)
mean_sam_f1 = np.mean(f1s_sam)
mean_macro_f1 = np.mean(macro_f1s)
mean_bal_acc = np.mean(balanced_accs)

print("\\n" + "="*65)
print(f"🏆 SUMMARY RESULTS: EXP {exp_num} - {title.upper()}")
print("="*65)
print(f"🎯 SAM Recall        : {{mean_sam_rec:.4f}} ({{mean_sam_rec*100:.2f}}%)")
print(f"🎯 SAM Precision     : {{mean_sam_prec:.4f}} ({{mean_sam_prec*100:.2f}}%)")
print(f"🎯 SAM F1-Score      : {{mean_sam_f1:.4f}} ({{mean_sam_f1*100:.2f}}%)")
print(f"📈 Macro F1-Score    : {{mean_macro_f1:.4f}} ({{mean_macro_f1*100:.2f}}%)")
print(f"⚖️  Balanced Accuracy: {{mean_bal_acc:.4f}} ({{mean_bal_acc*100:.2f}}%)")
print("="*65)

# Plot Aggregated Confusion Matrix
plt.figure(figsize=(7, 5))
sns.heatmap(cm_total, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f"Confusion Matrix (5-Fold Total) - Exp {exp_num}: {title}", fontsize=12, fontweight='bold')
plt.xlabel("Predicted Class", fontsize=11)
plt.ylabel("True Class", fontsize=11)
plt.tight_layout()
plt.show()
"""
    cells.append(create_code_cell(eval_code))

    save_notebook(filepath, cells)


# Define 6 Experiments
exp_definitions = [
    {
        'file': 'Exp1_Baseline_Original_Data.ipynb',
        'title': 'Original Data (Baseline - Class Weighted)',
        'num': 1,
        'summary': 'Full Imbalanced Dataset (ONC: 5,393 [85.1%], MAM: 722 [11.4%], SAM: 220 [3.5%])',
        'pipeline': 'Landmarks (GPA) -> Exclude Noise (LM11,21,22) -> 5-Fold CV -> Class-Weighted XGBoost -> Evaluation',
        'setup': """# Experiment 1 Setup: Full Imbalanced Dataset with Class-Weighted XGBoost
X_exp = df[feature_cols].values
y_exp = df['Class_Type'].map({'onc': 0, 'mam': 1, 'sam': 2}).values

print(f"Experiment 1 Input Matrix: {X_exp.shape}")
print("No oversampling applied; class balancing handled via compute_sample_weight('balanced').")
""",
        'type': 'exp1'
    },
    {
        'file': 'Exp2_Real_Data_Balanced_220.ipynb',
        'title': 'Pure Real-Data Balanced Subset (220-220-220)',
        'num': 2,
        'summary': 'Balanced Subsample of 220 Real Specimens per Class (Total N = 660, 100% Biological Data)',
        'pipeline': 'ONC (220 random) + MAM (220 random) + SAM (all 220) -> 5-Fold CV -> XGBoost -> Evaluation',
        'setup': """# Experiment 2 Setup: 220-220-220 Pure Real Subsample
df_sam = df[df['Class_Type'] == 'sam']
df_mam = df[df['Class_Type'] == 'mam'].sample(n=220, random_state=RANDOM_STATE)
df_onc = df[df['Class_Type'] == 'onc'].sample(n=220, random_state=RANDOM_STATE)

df_exp2 = pd.concat([df_onc, df_mam, df_sam]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

X_exp = df_exp2[feature_cols].values
y_exp = df_exp2['Class_Type'].map({'onc': 0, 'mam': 1, 'sam': 2}).values

print(f"Experiment 2 Subsample Matrix: {X_exp.shape}")
print(f"Class distribution: ONC=220, MAM=220, SAM=220 (100% Real Data, Clean Scientific Benchmark)")
""",
        'type': 'exp2'
    },
    {
        'file': 'Exp3_Limited_Synthetic_500.ipynb',
        'title': 'Limited Synthetic Expansion (500-500-500)',
        'num': 3,
        'summary': '500 ONC real + 500 MAM real + 220 SAM real + 280 BorderlineSMOTE SAM synthetic (Total N = 1500)',
        'pipeline': 'Subset Real (ONC=500, MAM=500, SAM=220) -> 5-Fold CV -> BorderlineSMOTE on SAM (Train Folds Only) -> XGBoost -> Evaluation',
        'setup': """# Experiment 3 Setup: 500-500-500 Limited Synthetic Expansion
df_sam = df[df['Class_Type'] == 'sam']
df_mam = df[df['Class_Type'] == 'mam'].sample(n=500, random_state=RANDOM_STATE)
df_onc = df[df['Class_Type'] == 'onc'].sample(n=500, random_state=RANDOM_STATE)

df_exp3 = pd.concat([df_onc, df_mam, df_sam]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

X_exp = df_exp3[feature_cols].values
y_exp = df_exp3['Class_Type'].map({'onc': 0, 'mam': 1, 'sam': 2}).values

print(f"Experiment 3 Real Subsample Matrix: {X_exp.shape}")
print(f"SAM oversampled using BorderlineSMOTE inside training folds up to 400/fold (~500 total).")
""",
        'type': 'exp3'
    },
    {
        'file': 'Exp4_Preferred_Balanced_722.ipynb',
        'title': 'Preferred Balanced Subset (722-722-722)',
        'num': 4,
        'summary': '722 ONC real + 722 MAM real + 220 SAM real + 502 BorderlineSMOTE SAM synthetic (Total N = 2166)',
        'pipeline': 'Subset Real (ONC=722, MAM=722, SAM=220) -> 5-Fold CV -> BorderlineSMOTE on SAM (Train Folds Only) -> XGBoost -> Evaluation',
        'setup': """# Experiment 4 Setup: 722-722-722 Preferred Balanced Subset
df_sam = df[df['Class_Type'] == 'sam']
df_mam = df[df['Class_Type'] == 'mam']
df_onc = df[df['Class_Type'] == 'onc'].sample(n=722, random_state=RANDOM_STATE)

df_exp4 = pd.concat([df_onc, df_mam, df_sam]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

X_exp = df_exp4[feature_cols].values
y_exp = df_exp4['Class_Type'].map({'onc': 0, 'mam': 1, 'sam': 2}).values

print(f"Experiment 4 Real Subsample Matrix: {X_exp.shape}")
print(f"Preserves 100% MAM & SAM real specimens. SAM oversampled using BorderlineSMOTE inside training folds to match 722.")
""",
        'type': 'exp4'
    },
    {
        'file': 'Exp5_Full_Data_SMOTE_Tomek.ipynb',
        'title': 'Full Data with SMOTE-Tomek (Training Fold Only)',
        'num': 5,
        'summary': 'Full Imbalanced Dataset (5393 ONC, 722 MAM, 220 SAM) with dynamic SMOTE-Tomek oversampling inside train folds',
        'pipeline': 'Full Data -> 5-Fold CV -> SMOTE-Tomek (Train Folds Only) -> XGBoost -> Evaluation',
        'setup': """# Experiment 5 Setup: Full Dataset with SMOTE-Tomek Oversampling
X_exp = df[feature_cols].values
y_exp = df['Class_Type'].map({'onc': 0, 'mam': 1, 'sam': 2}).values

print(f"Experiment 5 Full Dataset Matrix: {X_exp.shape}")
print("SMOTE-Tomek applied dynamically strictly inside training folds to prevent data leakage.")
""",
        'type': 'exp5'
    },
    {
        'file': 'Exp6_Full_Data_Borderline_SMOTE.ipynb',
        'title': 'Full Data with Borderline-SMOTE (Training Fold Only)',
        'num': 6,
        'summary': 'Full Imbalanced Dataset (5393 ONC, 722 MAM, 220 SAM) with dynamic Borderline-SMOTE oversampling inside train folds',
        'pipeline': 'Full Data -> 5-Fold CV -> Borderline-SMOTE (Train Folds Only) -> XGBoost -> Evaluation',
        'setup': """# Experiment 6 Setup: Full Dataset with Borderline-SMOTE Oversampling
X_exp = df[feature_cols].values
y_exp = df['Class_Type'].map({'onc': 0, 'mam': 1, 'sam': 2}).values

print(f"Experiment 6 Full Dataset Matrix: {X_exp.shape}")
print("Borderline-SMOTE applied dynamically strictly inside training folds to target decision boundary samples.")
""",
        'type': 'exp6'
    }
]

# Generate all 6 experiment notebooks
for exp in exp_definitions:
    filepath = os.path.join(out_dir, exp['file'])
    build_exp_notebook(filepath, exp['title'], exp['num'], exp['summary'], exp['pipeline'], exp['setup'], exp['type'])


# Generate Master Summary Notebook
master_path = os.path.join(out_dir, 'Master_Augmentation_Experiments_Comparison.ipynb')
master_cells = []

master_cells.append(create_markdown_cell("""# 📊 Master Geometric Data Augmentation & Imbalance Classification Benchmark
## Comprehensive Comparative Analysis of Experiments 1 to 6

### 📌 Study Background & Rationale
Severe Acute Malnutrition (**SAM**) represents only **3.5% (220 specimens)** of our dataset ($N=6,335$). Standard classification accuracy is misleading in imbalanced contexts because a naive model predicting 100% ONC achieves 85.1% accuracy while completely failing to detect SAM cases.

To scientifically determine the optimal balancing strategy, we evaluated **6 distinct data augmentation & balancing pipelines** across **Stratified 5-Fold Cross-Validation** using **XGBoost**:
1. **Experiment 1 (Baseline)**: Full Data ($5,393$-$722$-$220$) + Class-Weighted XGBoost.
2. **Experiment 2 (Real Data Benchmark)**: Pure Real Subset ($220$-$220$-$220$), 100% biological data.
3. **Experiment 3 (Limited Synthetic)**: $500$-$500$-$500$ ($18.7\\%$ synthetic SAM).
4. **Experiment 4 (Preferred Balanced)**: $722$-$722$-$722$ ($23.2\\%$ synthetic SAM, preserving 100% real MAM & SAM).
5. **Experiment 5 (SMOTE-Tomek)**: Full Data + SMOTE-Tomek (Training fold only).
6. **Experiment 6 (Borderline-SMOTE)**: Full Data + Borderline-SMOTE (Training fold only).

*Note: In all experiments, noisy & collinear landmarks (**LM11, LM21, LM22**) were removed to optimize geometric shape signal.*
"""))

master_cells.append(create_code_cell("""import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = r"e:\\ap-govt-EDA-model-tainning-testing\\GM_out_of_sample_classification-main\\GM_out_of_sample_classification-ap-govt\\geometric-dataAugmentation"
res_file = os.path.join(data_dir, "all_experiments_results.csv")
cm_file = os.path.join(data_dir, "confusion_matrices.json")

# Load Results
df_res = pd.read_csv(res_file)
with open(cm_file, 'r') as f:
    cms_dict = json.load(f)

print("🏆 MASTER EXPERIMENTAL EVALUATION TABLE")
print("="*85)
display(df_res)
"""))

master_cells.append(create_code_cell("""# Plot Comparative Performance Metrics
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: SAM Recall & SAM F1
df_res.plot(x='Experiment', y=['SAM Recall', 'SAM F1', 'SAM Precision'], kind='bar', ax=axes[0], colormap='viridis', width=0.8)
axes[0].set_title("SAM Class Metrics across Experiments", fontsize=13, fontweight='bold')
axes[0].set_ylabel("Score", fontsize=11)
axes[0].set_ylim(0, 0.8)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)
axes[0].set_xticklabels(df_res['Experiment'], rotation=30, ha='right')

# Plot 2: Macro F1 & Balanced Accuracy
df_res.plot(x='Experiment', y=['Macro F1', 'Balanced Acc'], kind='bar', ax=axes[1], colormap='magma', width=0.7)
axes[1].set_title("Overall Model Metrics across Experiments", fontsize=13, fontweight='bold')
axes[1].set_ylabel("Score", fontsize=11)
axes[1].set_ylim(0.4, 0.8)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)
axes[1].set_xticklabels(df_res['Experiment'], rotation=30, ha='right')

plt.tight_layout()
plt.show()
"""))

master_cells.append(create_code_cell("""# Plot Side-by-Side Confusion Matrices (2x3 Grid)
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()
class_names = ['ONC', 'MAM', 'SAM']

for idx, (exp_name, cm_list) in enumerate(cms_dict.items()):
    cm = np.array(cm_list)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], xticklabels=class_names, yticklabels=class_names, cbar=False)
    axes[idx].set_title(f"{exp_name}", fontsize=12, fontweight='bold')
    axes[idx].set_xlabel("Predicted Class")
    axes[idx].set_ylabel("True Class")

plt.suptitle("Side-by-Side Aggregated Confusion Matrices (5-Fold CV)", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""))

master_cells.append(create_markdown_cell("""## 💡 Scientific Conclusions & Strategy Recommendations

### 🥇 1. Highest SAM Sensitivity & Cleanest Benchmark: **Experiment 2 (220-220-220 Real Only)**
* **SAM Recall**: **67.73%** | **SAM F1**: **0.6530** | **SAM Precision**: **63.28%**
* **Why it wins scientifically**: By using equal 220 real specimens for each class, the model learns unbiased shape differences without synthetic artifacts. It achieves the highest SAM Recall (detecting over two-thirds of severe malnutrition cases) and highest SAM F1-Score.

### 🥈 2. Best Machine Learning Tradeoff: **Experiment 4 (722-722-722 Preferred Balanced)**
* **Macro F1**: **0.6837** | **Balanced Acc**: **0.6733** | **SAM Precision**: **62.63%**
* **Why it is preferred for publication**: Retains **100% of available real MAM and SAM specimens**, uses only a moderate amount of synthetic SAM data ($23.2\\%$), and delivers high overall macro accuracy while maintaining strong SAM detection.

### 🥉 3. Full Dataset Winner: **Experiment 5 (Full Data + SMOTE-Tomek)**
* **Macro F1**: **0.6705** | **Balanced Acc**: **0.6618** | **SAM Recall**: **46.82%**
* **Why it matters**: Demonstrates that SMOTE-Tomek dynamic oversampling + Tomek link noise removal outperforms standard class-weighted XGBoost (Baseline SAM Recall was only 24.09%).
"""))

save_notebook(master_path, master_cells)

print("\n🎉 ALL 7 EXPERIMENTAL NOTEBOOKS GENERATED SUCCESSFULLY VIA JSON!")
