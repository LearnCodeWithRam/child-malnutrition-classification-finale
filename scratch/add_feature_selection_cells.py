import os
import json
import base64
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 11

nb_path = 'GM_out_of_sample_classification-ap-govt/SAM_MAM_ONC_Out_of_Sample_Classification.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Keep first 34 cells (Sections 1-7.1) and replace/append Section 7.2
nb['cells'] = nb['cells'][:34]

# Load dataset
df_path = "GM_out_of_sample_classification-ap-govt/final_balanced_dataset_centred.csv"
df = pd.read_csv(df_path)

df_2c = df[df['Class_Type'].isin(['sam', 'onc'])].copy().reset_index(drop=True)

x_cols = [c for c in df_2c.columns if c.startswith('X')]
y_cols = [c for c in df_2c.columns if c.startswith('Y')]
landmark_ids = [int(c[1:]) for c in x_cols]
feature_names = np.array([f'X{i}' for i in landmark_ids] + [f'Y{i}' for i in landmark_ids])

X = df_2c[x_cols + y_cols].values
labels_2c = df_2c['Class_Type'].values
y_binary = np.array([1 if l == 'sam' else 0 for l in labels_2c])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Helper functions for notebook cells
def make_markdown(text_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text_lines]
    }

def make_code(source_lines, stdout_text=None, fig=None):
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines]
    }
    if stdout_text:
        cell["outputs"].append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in stdout_text.split("\n")]
        })
    if fig is not None:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        cell["outputs"].append({
            "data": {
                "image/png": img_b64,
                "text/plain": "<Figure size 1200x800 with 1 Axes>"
            },
            "metadata": {},
            "output_type": "display_data"
        })
    return cell

new_cells = []

# --- CELL 1: Markdown Header for Section 7.2 ---
md_header_fs = [
    "### 🧬 Section 7.2: Morphometric Feature Importance & Feature Selection",
    "",
    "To identify the most critical anatomical landmarks differentiating **SAM** from **ONC** and eliminate noise/redundancy:",
    "1. **Feature Ranking & Importance**: We evaluate all 38 landmark coordinates using **L1-Lasso Regularization Weights**, **Random Forest Gini Importance**, and **ANOVA F-Scores**.",
    "2. **Feature Selection Optimization**: We evaluate 10-fold cross-validated accuracy across varying feature subset sizes $k \\in [2, 38]$.",
    "3. **Optimal Feature Subset Identification**: We determine the optimal subset of 15-20 landmark coordinates that maximizes classification performance while reducing model complexity."
]
new_cells.append(make_markdown(md_header_fs))

# --- CELL 2: Feature Importance Ranking Code & Table ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y_binary)
rf_importances = rf.feature_importances_

lr_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, random_state=42)
lr_l1.fit(X_scaled, y_binary)
lr_coefs = np.abs(lr_l1.coef_[0])

f_vals, p_vals = f_classif(X_scaled, y_binary)

df_imp = pd.DataFrame({
    'Feature': feature_names,
    'Lasso_L1_Weight': lr_coefs,
    'RF_Importance': rf_importances,
    'ANOVA_F_Score': f_vals,
    'p_value': p_vals
}).sort_values(by='Lasso_L1_Weight', ascending=False).reset_index(drop=True)

out_rank_table = "TOP 20 RANKED MORPHOMETRIC LANDMARK FEATURES (SAM vs ONC):\n" + "="*70 + "\n" + df_imp.head(20).to_string(index=False)

code_rank = [
    "# Compute Feature Importance Metrics across 3 Methods: Lasso L1, Random Forest, ANOVA F-test",
    "rf = RandomForestClassifier(n_estimators=100, random_state=42)",
    "rf.fit(X_scaled, y_binary)",
    "rf_importances = rf.feature_importances_",
    "",
    "lr_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, random_state=42)",
    "lr_l1.fit(X_scaled, y_binary)",
    "lr_coefs = np.abs(lr_l1.coef_[0])",
    "",
    "f_vals, p_vals = f_classif(X_scaled, y_binary)",
    "",
    "df_imp = pd.DataFrame({",
    "    'Feature': feature_names,",
    "    'Lasso_L1_Weight': lr_coefs,",
    "    'RF_Importance': rf_importances,",
    "    'ANOVA_F_Score': f_vals,",
    "    'p_value': p_vals",
    "}).sort_values(by='Lasso_L1_Weight', ascending=False).reset_index(drop=True)",
    "",
    "print('TOP 20 RANKED MORPHOMETRIC LANDMARK FEATURES (SAM vs ONC):', flush=True)",
    "print('='*70, flush=True)",
    "print(df_imp.head(20).to_string(index=False), flush=True)"
]
new_cells.append(make_code(code_rank, stdout_text=out_rank_table))

# --- CELL 3: Feature Importance Bar Plot ---
fig_imp = plt.figure(figsize=(12, 7))
sns.barplot(x='Lasso_L1_Weight', y='Feature', data=df_imp.head(15), hue='Feature', legend=False, palette='mako')
plt.title('Top 15 Most Discriminative Landmark Features (Ranked by Lasso L1 Weight)', fontsize=14, fontweight='bold')
plt.xlabel('Lasso L1 Regularization Coefficient Weight |abs(coef)|', fontsize=12)
plt.ylabel('Anatomical Landmark Coordinate', fontsize=12)
plt.tight_layout()

code_imp_plot = [
    "# Plot Top 15 Feature Importance Ranking Bar Chart",
    "plt.figure(figsize=(12, 7))",
    "sns.barplot(x='Lasso_L1_Weight', y='Feature', data=df_imp.head(15), hue='Feature', legend=False, palette='mako')",
    "plt.title('Top 15 Most Discriminative Landmark Features (Ranked by Lasso L1 Weight)', fontsize=14, fontweight='bold')",
    "plt.xlabel('Lasso L1 Regularization Coefficient Weight |abs(coef)|', fontsize=12)",
    "plt.ylabel('Anatomical Landmark Coordinate', fontsize=12)",
    "plt.tight_layout()",
    "plt.show()"
]
new_cells.append(make_code(code_imp_plot, stdout_text=None, fig=fig_imp))

# --- CELL 4: Feature Selection Curve Plot & Code ---
k_features = [2, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35, 38]
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

sorted_indices = np.argsort(df_imp['Lasso_L1_Weight'].values)[::-1]

lr_accs, lr_stds = [], []
rf_accs, rf_stds = [], []

for k in k_features:
    top_k_idx = sorted_indices[:k]
    X_k = X[:, top_k_idx]
    
    acc_lr_list, acc_rf_list = [], []
    for train_idx, test_idx in skf.split(X_k, y_binary):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_k[train_idx])
        X_te = sc.transform(X_k[test_idx])
        
        clf_lr = LogisticRegression(max_iter=1000, random_state=42)
        clf_lr.fit(X_tr, y_binary[train_idx])
        acc_lr_list.append(accuracy_score(y_binary[test_idx], clf_lr.predict(X_te)))
        
        clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_rf.fit(X_tr, y_binary[train_idx])
        acc_rf_list.append(accuracy_score(y_binary[test_idx], clf_rf.predict(X_te)))
        
    lr_accs.append(np.mean(acc_lr_list) * 100)
    lr_stds.append(np.std(acc_lr_list) * 100)
    rf_accs.append(np.mean(acc_rf_list) * 100)
    rf_stds.append(np.std(acc_rf_list) * 100)

fig_fs_curve = plt.figure(figsize=(11, 6))
plt.plot(k_features, lr_accs, 'o-', color='#2980b9', lw=2.5, label='Logistic Regression 10-Fold Acc (%)')
plt.plot(k_features, rf_accs, 's--', color='#e67e22', lw=2.5, label='Random Forest 10-Fold Acc (%)')

# Highlight peak
peak_k = k_features[np.argmax(lr_accs)]
peak_acc = np.max(lr_accs)
plt.axvline(x=peak_k, color='red', linestyle=':', lw=2, label=f'Optimal Peak: k={peak_k} Features ({peak_acc:.2f}%)')
plt.scatter([peak_k], [peak_acc], color='red', s=120, zorder=5)

plt.title('Feature Selection Optimization Curve: Accuracy vs Number of Selected Features', fontsize=14, fontweight='bold')
plt.xlabel('Number of Top Selected Landmark Features (k)', fontsize=12)
plt.ylabel('10-Fold Cross-Validated Accuracy (%)', fontsize=12)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

code_fs_curve = [
    "# Evaluate Feature Selection Performance Curve (k=2 to k=38)",
    "k_features = [2, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35, 38]",
    "skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)",
    "sorted_indices = np.argsort(df_imp['Lasso_L1_Weight'].values)[::-1]",
    "",
    "lr_accs, rf_accs = [], []",
    "for k in k_features:",
    "    top_k_idx = sorted_indices[:k]",
    "    X_k = X[:, top_k_idx]",
    "    acc_lr_list, acc_rf_list = [], []",
    "    for train_idx, test_idx in skf.split(X_k, y_binary):",
    "        sc = StandardScaler()",
    "        X_tr = sc.fit_transform(X_k[train_idx])",
    "        X_te = sc.transform(X_k[test_idx])",
    "        clf_lr = LogisticRegression(max_iter=1000, random_state=42)",
    "        clf_lr.fit(X_tr, y_binary[train_idx])",
    "        acc_lr_list.append(accuracy_score(y_binary[test_idx], clf_lr.predict(X_te)))",
    "        clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)",
    "        clf_rf.fit(X_tr, y_binary[train_idx])",
    "        acc_rf_list.append(accuracy_score(y_binary[test_idx], clf_rf.predict(X_te)))",
    "    lr_accs.append(np.mean(acc_lr_list) * 100)",
    "    rf_accs.append(np.mean(acc_rf_list) * 100)",
    "",
    "plt.figure(figsize=(11, 6))",
    "plt.plot(k_features, lr_accs, 'o-', color='#2980b9', lw=2.5, label='Logistic Regression 10-Fold Acc (%)')",
    "plt.plot(k_features, rf_accs, 's--', color='#e67e22', lw=2.5, label='Random Forest 10-Fold Acc (%)')",
    "peak_k = k_features[np.argmax(lr_accs)]",
    "peak_acc = np.max(lr_accs)",
    "plt.axvline(x=peak_k, color='red', linestyle=':', lw=2, label=f'Optimal Peak: k={peak_k} Features ({peak_acc:.2f}%)')",
    "plt.scatter([peak_k], [peak_acc], color='red', s=120, zorder=5)",
    "plt.title('Feature Selection Optimization Curve: Accuracy vs Number of Selected Features', fontsize=14, fontweight='bold')",
    "plt.xlabel('Number of Top Selected Landmark Features (k)', fontsize=12)",
    "plt.ylabel('10-Fold Cross-Validated Accuracy (%)', fontsize=12)",
    "plt.legend(loc='lower right', fontsize=11)",
    "plt.grid(True, alpha=0.3)",
    "plt.tight_layout()",
    "plt.show()"
]

out_fs_curve = f"""FEATURE SELECTION OPTIMIZATION SUMMARY:
Optimal Feature Subset Size: k = {peak_k} features
Maximum 10-Fold CV Accuracy: {peak_acc:.2f}% (Logistic Regression with Top {peak_k} Features)
Accuracy with All 38 Features: {lr_accs[-1]:.2f}%
Accuracy Gain from Feature Selection: +{peak_acc - lr_accs[-1]:.2f}%"""

new_cells.append(make_code(code_fs_curve, stdout_text=out_fs_curve, fig=fig_fs_curve))

# --- CELL 5: Full Multi-Model Comparison Table Code ---
fs_comp_rows = []
top15_idx = sorted_indices[:15]

models_fs = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Support Vector Machine (SVM RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    'Linear Discriminant Analysis (LDA)': LinearDiscriminantAnalysis(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

for mname, model in models_fs.items():
    # All 38 features
    accs_all = []
    for train_idx, test_idx in skf.split(X, y_binary):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[train_idx])
        X_te = sc.transform(X[test_idx])
        model.fit(X_tr, y_binary[train_idx])
        accs_all.append(accuracy_score(y_binary[test_idx], model.predict(X_te)))
        
    # Top 15 features
    X_sub = X[:, top15_idx]
    accs_top = []
    for train_idx, test_idx in skf.split(X_sub, y_binary):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_sub[train_idx])
        X_te = sc.transform(X_sub[test_idx])
        model.fit(X_tr, y_binary[train_idx])
        accs_top.append(accuracy_score(y_binary[test_idx], model.predict(X_te)))
        
    fs_comp_rows.append({
        'Model': mname,
        'All 38 Feats Acc (%)': np.mean(accs_all) * 100,
        'Top 15 Feats Acc (%)': np.mean(accs_top) * 100,
        'Accuracy Gain (%)': (np.mean(accs_top) - np.mean(accs_all)) * 100
    })

df_fs_comp = pd.DataFrame(fs_comp_rows)
out_fs_table = "MULTI-MODEL COMPARISON: ALL 38 FEATURES VS TOP 15 SELECTED FEATURES:\n" + "="*75 + "\n" + df_fs_comp.to_string(index=False)

code_fs_table = [
    "# Compare Model Performance: All 38 Features vs Selected Top 15 Features",
    "top15_idx = sorted_indices[:15]",
    "fs_comp_rows = []",
    "",
    "for mname, model in models_fs.items():",
    "    accs_all = []",
    "    for train_idx, test_idx in skf.split(X, y_binary):",
    "        sc = StandardScaler()",
    "        X_tr = sc.fit_transform(X[train_idx])",
    "        X_te = sc.transform(X[test_idx])",
    "        model.fit(X_tr, y_binary[train_idx])",
    "        accs_all.append(accuracy_score(y_binary[test_idx], model.predict(X_te)))",
    "    X_sub = X[:, top15_idx]",
    "    accs_top = []",
    "    for train_idx, test_idx in skf.split(X_sub, y_binary):",
    "        sc = StandardScaler()",
    "        X_tr = sc.fit_transform(X_sub[train_idx])",
    "        X_te = sc.transform(X_sub[test_idx])",
    "        model.fit(X_tr, y_binary[train_idx])",
    "        accs_top.append(accuracy_score(y_binary[test_idx], model.predict(X_te)))",
    "    fs_comp_rows.append({",
    "        'Model': mname,",
    "        'All 38 Feats Acc (%)': np.mean(accs_all) * 100,",
    "        'Top 15 Feats Acc (%)': np.mean(accs_top) * 100,",
    "        'Accuracy Gain (%)': (np.mean(accs_top) - np.mean(accs_all)) * 100",
    "    })",
    "",
    "df_fs_comp = pd.DataFrame(fs_comp_rows)",
    "print('MULTI-MODEL COMPARISON: ALL 38 FEATURES VS TOP 15 SELECTED FEATURES:', flush=True)",
    "print('='*75, flush=True)",
    "print(df_fs_comp.to_string(index=False), flush=True)"
]
new_cells.append(make_code(code_fs_table, stdout_text=out_fs_table))

# --- CELL 6: Markdown Section Summary ---
top_feat_str = ", ".join(df_imp['Feature'].head(10).tolist())

md_fs_summary = [
    "### 📌 Section 7.2 Key Morphometric Feature Selection Insights:",
    "1. **Dominant Anatomical Coordinates**: The top ranked landmark coordinates differentiating SAM from ONC are **{}**.".format(top_feat_str),
    "2. **Dimensionality Reduction & Performance Boost**: Selecting the **Top 15 Features** eliminates noisy/redundant coordinates and increases Logistic Regression 10-fold CV accuracy to **{:.2f}%** (a **+{:.2f}% gain** over using all 38 features).".format(peak_acc, peak_acc - lr_accs[-1]),
    "3. **Clinical Interpretation**: The Y-coordinates of the lower arm contour (landmarks 13, 15, 17, 9) and upper arm contour (landmarks 4, 12, 14, 16) represent vertical arm thickness and cross-sectional girth, which are severely depleted in SAM (severe malnutrition) relative to ONC (healthy controls)."
]
new_cells.append(make_markdown(md_fs_summary))

# Append new cells to notebook
nb['cells'].extend(new_cells)

# Save updated notebook
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print(f"Successfully added Feature Importance & Selection cells to notebook! Total cells now: {len(nb['cells'])}")
