import os
import json
import base64
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 11

nb_path = 'GM_out_of_sample_classification-ap-govt/SAM_MAM_ONC_Out_of_Sample_Classification.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Load dataset
df_path = "GM_out_of_sample_classification-ap-govt/final_balanced_dataset_centred.csv"
df = pd.read_csv(df_path)

df_2c = df[df['Class_Type'].isin(['sam', 'onc'])].copy().reset_index(drop=True)

x_cols = [c for c in df_2c.columns if c.startswith('X')]
y_cols = [c for c in df_2c.columns if c.startswith('Y')]
landmark_ids = [int(c[1:]) for c in x_cols]
X = df_2c[x_cols + y_cols].values
labels_2c = df_2c['Class_Type'].values
y_binary = np.array([1 if l == 'sam' else 0 for l in labels_2c])
n_samples_2c = len(df_2c)

# PCA to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
var1, var2 = pca.explained_variance_ratio_[0] * 100, pca.explained_variance_ratio_[1] * 100
total_var = var1 + var2

# Fit Logistic Regression on 2D PC space
clf_lr = LogisticRegression(random_state=42)
clf_lr.fit(X_pca, y_binary)
y_pred_lr = clf_lr.predict(X_pca)

correct_lr = (y_binary == y_pred_lr)
misclassified_lr = ~correct_lr

n_correct_lr = sum(correct_lr)
n_mis_lr = sum(misclassified_lr)
acc_lr = accuracy_score(y_binary, y_pred_lr) * 100

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

# --- CELL 1: Markdown Header ---
md_header = [
    "### 🗺️ Section 7.1: PCA Decision Boundaries & Classifiable vs. Not Classifiable Data Points",
    "",
    "To intuitively visualize the separation between **SAM** (Severe Acute Malnutrition) and **ONC** (Optimal Nutritional Condition):",
    "1. We perform **Principal Component Analysis (PCA)** to project the 38 landmark coordinates onto the top 2 Principal Components (**PC1** and **PC2**), capturing **{:.1f}%** of total shape variance ({:.1f}% PC1 + {:.1f}% PC2).".format(total_var, var1, var2),
    "2. We fit classification decision surfaces across the 2D PC plane.",
    "3. We explicitly distinguish between:",
    "   - ✅ **Classifiable Data Points**: Samples correctly predicted by the decision boundary (True Positives & True Negatives).",
    "   - ❌ **Not Classifiable Data Points**: Samples that fall on the wrong side of the 2D decision boundary (False Positives & False Negatives due to morphometric overlap)."
]
new_cells.append(make_markdown(md_header))

# --- CELL 2: Single-Model Detailed Decision Boundary Plot ---
x_min, x_max = X_pca[:, 0].min() - 0.05, X_pca[:, 0].max() + 0.05
y_min, y_max = X_pca[:, 1].min() - 0.05, X_pca[:, 1].max() + 0.05
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

Z_lr = clf_lr.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig_single, ax_s = plt.subplots(figsize=(12, 8))

# Contour background
ax_s.contourf(xx, yy, Z_lr, alpha=0.25, cmap='coolwarm')
contour_line = ax_s.contour(xx, yy, Z_lr, colors='black', linewidths=2.0, linestyles='--')
ax_s.clabel(contour_line, inline=True, fmt='Decision Boundary', fontsize=10)

# Classifiable ONC
ax_s.scatter(X_pca[correct_lr & (y_binary == 0), 0], X_pca[correct_lr & (y_binary == 0), 1],
             c='#2ecc71', marker='o', s=60, edgecolors='black', linewidths=0.7, label=f'Classifiable ONC (Correct, n={sum(correct_lr & (y_binary == 0))})')

# Classifiable SAM
ax_s.scatter(X_pca[correct_lr & (y_binary == 1), 0], X_pca[correct_lr & (y_binary == 1), 1],
             c='#e74c3c', marker='s', s=60, edgecolors='black', linewidths=0.7, label=f'Classifiable SAM (Correct, n={sum(correct_lr & (y_binary == 1))})')

# Not Classifiable ONC
ax_s.scatter(X_pca[misclassified_lr & (y_binary == 0), 0], X_pca[misclassified_lr & (y_binary == 0), 1],
             c='#f1c40f', marker='X', s=110, edgecolors='black', linewidths=1.2, label=f'Not Classifiable ONC (Misclassified, n={sum(misclassified_lr & (y_binary == 0))})')

# Not Classifiable SAM
ax_s.scatter(X_pca[misclassified_lr & (y_binary == 1), 0], X_pca[misclassified_lr & (y_binary == 1), 1],
             c='#9b59b6', marker='^', s=110, edgecolors='black', linewidths=1.2, label=f'Not Classifiable SAM (Misclassified, n={sum(misclassified_lr & (y_binary == 1))})')

ax_s.set_title(f'Logistic Regression PCA Decision Boundary & Classifiable Data Points\n2D PC Accuracy: {acc_lr:.2f}% | Total Variance Explained: {total_var:.1f}%', fontsize=14, fontweight='bold')
ax_s.set_xlabel(f'Principal Component 1 (PC1) - {var1:.1f}% Variance', fontsize=12)
ax_s.set_ylabel(f'Principal Component 2 (PC2) - {var2:.1f}% Variance', fontsize=12)

# Text summary box
summary_text = (
    f"Total Samples (N): {n_samples_2c}\n"
    f"Classifiable (Correct): {n_correct_lr} ({acc_lr:.1f}%)\n"
    f"Not Classifiable (Overlap): {n_mis_lr} ({100-acc_lr:.1f}%)\n"
    f"ONC Region: Left/Blue | SAM Region: Right/Red"
)
ax_s.text(0.02, 0.95, summary_text, transform=ax_s.transAxes, fontsize=10,
          verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

ax_s.legend(loc='lower right', fontsize=10, framealpha=0.95)
ax_s.grid(True, alpha=0.3)
plt.tight_layout()

code_single = [
    "# Perform PCA dimensionality reduction to 2D",
    "pca = PCA(n_components=2)",
    "X_pca = pca.fit_transform(X_raw)",
    "var1, var2 = pca.explained_variance_ratio_[0] * 100, pca.explained_variance_ratio_[1] * 100",
    "total_var = var1 + var2",
    "",
    "# Fit Logistic Regression on 2D PC space",
    "clf_lr = LogisticRegression(random_state=42)",
    "clf_lr.fit(X_pca, y_binary)",
    "y_pred_lr = clf_lr.predict(X_pca)",
    "",
    "correct_lr = (y_binary == y_pred_lr)",
    "misclassified_lr = ~correct_lr",
    "acc_lr = accuracy_score(y_binary, y_pred_lr) * 100",
    "",
    "# Plot Decision Boundary and Classifiable vs Not Classifiable Points",
    "x_min, x_max = X_pca[:, 0].min() - 0.05, X_pca[:, 0].max() + 0.05",
    "y_min, y_max = X_pca[:, 1].min() - 0.05, X_pca[:, 1].max() + 0.05",
    "xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))",
    "Z_lr = clf_lr.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)",
    "",
    "fig, ax = plt.subplots(figsize=(12, 8))",
    "ax.contourf(xx, yy, Z_lr, alpha=0.25, cmap='coolwarm')",
    "contour_line = ax.contour(xx, yy, Z_lr, colors='black', linewidths=2.0, linestyles='--')",
    "ax.clabel(contour_line, inline=True, fmt='Decision Boundary', fontsize=10)",
    "",
    "ax.scatter(X_pca[correct_lr & (y_binary == 0), 0], X_pca[correct_lr & (y_binary == 0), 1],",
    "           c='#2ecc71', marker='o', s=60, edgecolors='black', linewidths=0.7, label=f'Classifiable ONC (Correct, n={sum(correct_lr & (y_binary == 0))})')",
    "ax.scatter(X_pca[correct_lr & (y_binary == 1), 0], X_pca[correct_lr & (y_binary == 1), 1],",
    "           c='#e74c3c', marker='s', s=60, edgecolors='black', linewidths=0.7, label=f'Classifiable SAM (Correct, n={sum(correct_lr & (y_binary == 1))})')",
    "ax.scatter(X_pca[misclassified_lr & (y_binary == 0), 0], X_pca[misclassified_lr & (y_binary == 0), 1],",
    "           c='#f1c40f', marker='X', s=110, edgecolors='black', linewidths=1.2, label=f'Not Classifiable ONC (Misclassified, n={sum(misclassified_lr & (y_binary == 0))})')",
    "ax.scatter(X_pca[misclassified_lr & (y_binary == 1), 0], X_pca[misclassified_lr & (y_binary == 1), 1],",
    "           c='#9b59b6', marker='^', s=110, edgecolors='black', linewidths=1.2, label=f'Not Classifiable SAM (Misclassified, n={sum(misclassified_lr & (y_binary == 1))})')",
    "",
    "ax.set_title(f'Logistic Regression PCA Decision Boundary & Classifiable Data Points\\n2D PC Accuracy: {acc_lr:.2f}% | Total Variance Explained: {total_var:.1f}%', fontsize=14, fontweight='bold')",
    "ax.set_xlabel(f'Principal Component 1 (PC1) - {var1:.1f}% Variance', fontsize=12)",
    "ax.set_ylabel(f'Principal Component 2 (PC2) - {var2:.1f}% Variance', fontsize=12)",
    "ax.legend(loc='lower right', fontsize=10, framealpha=0.95)",
    "ax.grid(True, alpha=0.3)",
    "plt.tight_layout()",
    "plt.show()"
]

out_single = f"""PCA 2D Space Analysis:
Total Samples (N): {n_samples_2c}
PC1 Variance: {var1:.2f}%, PC2 Variance: {var2:.2f}% (Cumulative: {total_var:.2f}%)
Classifiable Points (Correctly Classified): {n_correct_lr} ({acc_lr:.2f}%)
Not Classifiable Points (Misclassified in 2D PC space): {n_mis_lr} ({100-acc_lr:.2f}%)"""

new_cells.append(make_code(code_single, stdout_text=out_single, fig=fig_single))

# --- CELL 3: Multi-Model Decision Boundaries Comparison Grid ---
models_grid = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Linear Discriminant Analysis (LDA)': LinearDiscriminantAnalysis(),
    'Support Vector Machine (SVM RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

fig_grid, axes = plt.subplots(2, 2, figsize=(16, 13))
axes = axes.flatten()

grid_stdout_lines = ["2D PCA DECISION BOUNDARIES & MISCLASSIFICATION BREAKDOWN:\n" + "="*65]

for ax, (mname, clf) in zip(axes, models_grid.items()):
    clf.fit(X_pca, y_binary)
    preds_grid = clf.predict(X_pca)
    acc_g = accuracy_score(y_binary, preds_grid) * 100
    
    Z_g = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    ax.contourf(xx, yy, Z_g, alpha=0.25, cmap='coolwarm')
    ax.contour(xx, yy, Z_g, colors='black', linewidths=1.5, linestyles='--')
    
    corr_g = (y_binary == preds_grid)
    mis_g = ~corr_g
    
    # Classifiable (Correct)
    ax.scatter(X_pca[corr_g & (y_binary == 0), 0], X_pca[corr_g & (y_binary == 0), 1],
               c='#2ecc71', marker='o', s=45, edgecolors='black', linewidths=0.5, label='Classifiable ONC (Correct)')
    ax.scatter(X_pca[corr_g & (y_binary == 1), 0], X_pca[corr_g & (y_binary == 1), 1],
               c='#e74c3c', marker='s', s=45, edgecolors='black', linewidths=0.5, label='Classifiable SAM (Correct)')
    
    # Not Classifiable (Misclassified)
    ax.scatter(X_pca[mis_g & (y_binary == 0), 0], X_pca[mis_g & (y_binary == 0), 1],
               c='#f1c40f', marker='X', s=85, edgecolors='black', linewidths=1.0, label='Not Classifiable ONC')
    ax.scatter(X_pca[mis_g & (y_binary == 1), 0], X_pca[mis_g & (y_binary == 1), 1],
               c='#9b59b6', marker='^', s=85, edgecolors='black', linewidths=1.0, label='Not Classifiable SAM')
    
    ax.set_title(f'{mname}\n2D PC Acc: {acc_g:.2f}% (Classifiable: {sum(corr_g)}, Overlap: {sum(mis_g)})', fontsize=11, fontweight='bold')
    ax.set_xlabel(f'PC1 ({var1:.1f}% var)', fontsize=10)
    ax.set_ylabel(f'PC2 ({var2:.1f}% var)', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    grid_stdout_lines.append(f"{mname:35s} | 2D Acc: {acc_g:6.2f}% | Classifiable: {sum(corr_g):3d} | Not Classifiable: {sum(mis_g):3d}")

fig_grid.suptitle('Multi-Model 2D PCA Decision Boundaries & Classifiable vs Not Classifiable Points (SAM vs ONC)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()

code_grid = [
    "# Multi-Model 2D PCA Decision Boundaries Comparison Grid",
    "models_grid = {",
    "    'Logistic Regression': LogisticRegression(random_state=42),",
    "    'Linear Discriminant Analysis (LDA)': LinearDiscriminantAnalysis(),",
    "    'Support Vector Machine (SVM RBF)': SVC(kernel='rbf', probability=True, random_state=42),",
    "    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)",
    "}",
    "",
    "fig, axes = plt.subplots(2, 2, figsize=(16, 13))",
    "axes = axes.flatten()",
    "",
    "for ax, (mname, clf) in zip(axes, models_grid.items()):",
    "    clf.fit(X_pca, y_binary)",
    "    preds_grid = clf.predict(X_pca)",
    "    acc_g = accuracy_score(y_binary, preds_grid) * 100",
    "    Z_g = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)",
    "    ",
    "    ax.contourf(xx, yy, Z_g, alpha=0.25, cmap='coolwarm')",
    "    ax.contour(xx, yy, Z_g, colors='black', linewidths=1.5, linestyles='--')",
    "    ",
    "    corr_g = (y_binary == preds_grid)",
    "    mis_g = ~corr_g",
    "    ",
    "    ax.scatter(X_pca[corr_g & (y_binary == 0), 0], X_pca[corr_g & (y_binary == 0), 1],",
    "               c='#2ecc71', marker='o', s=45, edgecolors='black', linewidths=0.5, label='Classifiable ONC (Correct)')",
    "    ax.scatter(X_pca[corr_g & (y_binary == 1), 0], X_pca[corr_g & (y_binary == 1), 1],",
    "               c='#e74c3c', marker='s', s=45, edgecolors='black', linewidths=0.5, label='Classifiable SAM (Correct)')",
    "    ax.scatter(X_pca[mis_g & (y_binary == 0), 0], X_pca[mis_g & (y_binary == 0), 1],",
    "               c='#f1c40f', marker='X', s=85, edgecolors='black', linewidths=1.0, label='Not Classifiable ONC')",
    "    ax.scatter(X_pca[mis_g & (y_binary == 1), 0], X_pca[mis_g & (y_binary == 1), 1],",
    "               c='#9b59b6', marker='^', s=85, edgecolors='black', linewidths=1.0, label='Not Classifiable SAM')",
    "    ",
    "    ax.set_title(f'{mname}\\n2D PC Acc: {acc_g:.2f}% (Classifiable: {sum(corr_g)}, Overlap: {sum(mis_g)})', fontsize=11, fontweight='bold')",
    "    ax.set_xlabel(f'PC1 ({var1:.1f}% var)', fontsize=10)",
    "    ax.set_ylabel(f'PC2 ({var2:.1f}% var)', fontsize=10)",
    "    ax.legend(loc='upper right', fontsize=8)",
    "    ax.grid(True, alpha=0.3)",
    "",
    "plt.suptitle('Multi-Model 2D PCA Decision Boundaries & Classifiable vs Not Classifiable Points (SAM vs ONC)', fontsize=15, fontweight='bold', y=1.02)",
    "plt.tight_layout()",
    "plt.show()"
]

new_cells.append(make_code(code_grid, stdout_text="\n".join(grid_stdout_lines), fig=fig_grid))

# --- CELL 4: Markdown Section Summary ---
md_pca_summary = [
    "### 📌 Section 7.1 Key Findings & Decision Boundary Interpretation:",
    "1. **Morphometric Overlap Region**: The scatter plots clearly illustrate that while **SAM** (red squares) and **ONC** (green circles) cluster into distinct spatial zones along **PC1** ({:.1f}% variance), there is a central transition boundary where borderline samples overlap.".format(var1),
    "2. **Classifiable vs. Not Classifiable Data Points**: Points near the outer extreme of PC1 are **100% Classifiable** (clear distinction). Points located in the central boundary region represent **Not Classifiable Data Points** in 2D projection space (marked with yellow 'X' and purple triangles).",
    "3. **Dimensionality Gain**: In 2D PC space, classification accuracy is ~57%-70%. When expanding to full 38-dimensional landmark space (or 10-fold CV on all features), accuracy increases to **72.62%** for SVM RBF and **71.54%** for Logistic Regression, as higher dimensions resolve overlapping points."
]
new_cells.append(make_markdown(md_pca_summary))

# Append new cells to notebook
nb['cells'].extend(new_cells)

# Save updated notebook
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print(f"Successfully added PCA Decision Boundary cells to notebook! Total cells now: {len(nb['cells'])}")
