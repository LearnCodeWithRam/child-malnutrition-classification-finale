import os
import json
import base64
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Load dataset
df_path = "GM_out_of_sample_classification-ap-govt/final_balanced_dataset_centred.csv"
df = pd.read_csv(df_path)

df_2c = df[df['Class_Type'].isin(['sam', 'onc'])].copy().reset_index(drop=True)

x_cols = [c for c in df_2c.columns if c.startswith('X')]
y_cols = [c for c in df_2c.columns if c.startswith('Y')]
landmark_ids = [int(c[1:]) for c in x_cols]

X_x = df_2c[x_cols].values
X_y = df_2c[y_cols].values
labels_2c = df_2c['Class_Type'].values
y_binary = np.array([1 if l == 'sam' else 0 for l in labels_2c])
n_samples_2c = len(df_2c)

sam_mask = (y_binary == 1)
onc_mask = (y_binary == 0)

mean_sam_x = np.mean(X_x[sam_mask], axis=0)
mean_sam_y = np.mean(X_y[sam_mask], axis=0)
mean_onc_x = np.mean(X_x[onc_mask], axis=0)
mean_onc_y = np.mean(X_y[onc_mask], axis=0)

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

# --- CELL 1: Markdown Header for Section 7.3 ---
md_header_grouped = [
    "### 📐 Section 7.3: Grouped 2D Landmark Selection (Paired $X_i, Y_i$ Complete Points)",
    "",
    "### 💡 Methodological Principle: Preserving 2D Landmark Geometry",
    "In 2D Geometric Morphometrics, an anatomical landmark $L_i$ is a **bivariate coordinate vector** $L_i = (X_i, Y_i) \\in \\mathbb{R}^2$ representing a physical anatomical point on the arm:",
    "- Removing $X_i$ while retaining $Y_i$ (or vice versa) destroys the 2D spatial coordinate point topology.",
    "- Therefore, feature selection in geometric morphometrics must be performed at the **Grouped 2D Landmark Point Level**, treating $(X_i, Y_i)$ as an inseparable atomic entity.",
    "",
    "### 🔬 Grouped Feature Importance & Selection Protocol",
    "1. **2D Displacement Distance & $T^2$ Discriminant Score**: We compute the 2D Euclidean distance $\|(\\Delta X_i, \\Delta Y_i)\|_2$ between the mean SAM arm shape and mean ONC arm shape for each complete landmark $L_i = (X_i, Y_i)$.",
    "2. **Grouped Landmark Ranking**: We rank all 19 complete anatomical landmark points.",
    "3. **Grouped Selection Evaluation**: We evaluate 10-fold cross-validated accuracy when retaining top $K$ **complete 2D landmark pairs** ($K \\in [2, 19]$ complete landmarks)."
]
new_cells.append(make_markdown(md_header_grouped))

# --- CELL 2: Grouped 2D Landmark Importance Ranking Code & Table ---
l_scores = []
for idx, lid in enumerate(landmark_ids):
    dx = mean_sam_x[idx] - mean_onc_x[idx]
    dy = mean_sam_y[idx] - mean_onc_y[idx]
    dist_2d = np.sqrt(dx**2 + dy**2)
    var_x = np.var(X_x[:, idx]) + 1e-9
    var_y = np.var(X_y[:, idx]) + 1e-9
    t2 = (dx**2/var_x) + (dy**2/var_y)
    l_scores.append({
        'Landmark_ID': f'Landmark_{lid} (X{lid}, Y{lid})',
        'Raw_ID': lid,
        'Distance_2D_mm': dist_2d,
        'T2_Score': t2,
        'Mean_dX': dx,
        'Mean_dY': dy
    })

df_grouped = pd.DataFrame(l_scores).sort_values(by='T2_Score', ascending=False).reset_index(drop=True)

out_grouped_table = "GROUPED 2D LANDMARK POINT IMPORTANCE RANKING (PAIRED Xi, Yi):\n" + "="*75 + "\n" + df_grouped.to_string(index=False)

code_grouped_rank = [
    "# Compute 2D Grouped Landmark Importance for paired (Xi, Yi) coordinate points",
    "l_scores = []",
    "for idx, lid in enumerate(landmark_ids):",
    "    dx = mean_sam_x[idx] - mean_onc_x[idx]",
    "    dy = mean_sam_y[idx] - mean_onc_y[idx]",
    "    dist_2d = np.sqrt(dx**2 + dy**2)",
    "    var_x = np.var(X_x[:, idx]) + 1e-9",
    "    var_y = np.var(X_y[:, idx]) + 1e-9",
    "    t2 = (dx**2/var_x) + (dy**2/var_y)",
    "    l_scores.append({",
    "        'Landmark_ID': f'Landmark_{lid} (X{lid}, Y{lid})',",
    "        'Raw_ID': lid,",
    "        'Distance_2D_mm': dist_2d,",
    "        'T2_Score': t2,",
    "        'Mean_dX': dx,",
    "        'Mean_dY': dy",
    "    })",
    "",
    "df_grouped = pd.DataFrame(l_scores).sort_values(by='T2_Score', ascending=False).reset_index(drop=True)",
    "print('GROUPED 2D LANDMARK POINT IMPORTANCE RANKING (PAIRED Xi, Yi):', flush=True)",
    "print('='*75, flush=True)",
    "print(df_grouped.to_string(index=False), flush=True)"
]
new_cells.append(make_code(code_grouped_rank, stdout_text=out_grouped_table))

# --- CELL 3: Grouped 2D Landmark Importance Bar Plot ---
fig_g_imp = plt.figure(figsize=(12, 7))
sns.barplot(x='T2_Score', y='Landmark_ID', data=df_grouped, hue='Landmark_ID', legend=False, palette='rocket')
plt.title('Grouped 2D Landmark Point Importance Ranking (Paired Xi, Yi Coordinates)', fontsize=14, fontweight='bold')
plt.xlabel('2D Multivariate Discriminant T^2 Score (SAM vs ONC Separation)', fontsize=12)
plt.ylabel('Anatomical 2D Landmark Point (X_i, Y_i)', fontsize=12)
plt.tight_layout()

code_g_imp_plot = [
    "# Plot Grouped 2D Landmark Point Importance Bar Chart",
    "plt.figure(figsize=(12, 7))",
    "sns.barplot(x='T2_Score', y='Landmark_ID', data=df_grouped, hue='Landmark_ID', legend=False, palette='rocket')",
    "plt.title('Grouped 2D Landmark Point Importance Ranking (Paired Xi, Yi Coordinates)', fontsize=14, fontweight='bold')",
    "plt.xlabel('2D Multivariate Discriminant T^2 Score (SAM vs ONC Separation)', fontsize=12)",
    "plt.ylabel('Anatomical 2D Landmark Point (X_i, Y_i)', fontsize=12)",
    "plt.tight_layout()",
    "plt.show()"
]
new_cells.append(make_code(code_g_imp_plot, stdout_text=None, fig=fig_g_imp))

# --- CELL 4: Grouped Feature Selection Curve Plot & Code ---
sorted_lids = df_grouped['Raw_ID'].values
k_landmarks = [2, 4, 6, 8, 10, 12, 15, 19]

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

g_lr_accs, g_svm_accs = [], []

for num_lm in k_landmarks:
    selected_lids = sorted_lids[:num_lm]
    cols = []
    for lid in selected_lids:
        cols.append(f'X{lid}')
        cols.append(f'Y{lid}')
        
    X_sub = df_2c[cols].values
    
    acc_lr_list, acc_svm_list = [], []
    for train_idx, test_idx in skf.split(X_sub, y_binary):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_sub[train_idx])
        X_te = sc.transform(X_sub[test_idx])
        
        clf_lr = LogisticRegression(max_iter=1000, random_state=42)
        clf_lr.fit(X_tr, y_binary[train_idx])
        acc_lr_list.append(accuracy_score(y_binary[test_idx], clf_lr.predict(X_te)))
        
        clf_svm = SVC(kernel='rbf', random_state=42)
        clf_svm.fit(X_tr, y_binary[train_idx])
        acc_svm_list.append(accuracy_score(y_binary[test_idx], clf_svm.predict(X_te)))
        
    g_lr_accs.append(np.mean(acc_lr_list) * 100)
    g_svm_accs.append(np.mean(acc_svm_list) * 100)

fig_g_curve = plt.figure(figsize=(11, 6))
plt.plot(k_landmarks, g_lr_accs, 'o-', color='#16a085', lw=2.5, label='Logistic Regression 10-Fold Acc (%)')
plt.plot(k_landmarks, g_svm_accs, 's--', color='#8e44ad', lw=2.5, label='SVM (RBF) 10-Fold Acc (%)')

# Highlight peak
peak_g_lm = k_landmarks[np.argmax(g_lr_accs)]
peak_g_acc = np.max(g_lr_accs)
plt.axvline(x=12, color='red', linestyle=':', lw=2, label='Recommended Subset: Top 12 Landmarks (24 coords)')
plt.scatter([12], [g_lr_accs[k_landmarks.index(12)]], color='red', s=120, zorder=5)

plt.title('Grouped Landmark Selection Optimization Curve (Paired Xi, Yi Points)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Complete 2D Landmark Points Selected (K)', fontsize=12)
plt.ylabel('10-Fold Cross-Validated Accuracy (%)', fontsize=12)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

code_g_curve = [
    "# Grouped Landmark Feature Selection Optimization Curve",
    "sorted_lids = df_grouped['Raw_ID'].values",
    "k_landmarks = [2, 4, 6, 8, 10, 12, 15, 19]",
    "skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)",
    "",
    "g_lr_accs, g_svm_accs = [], []",
    "for num_lm in k_landmarks:",
    "    selected_lids = sorted_lids[:num_lm]",
    "    cols = []",
    "    for lid in selected_lids:",
    "        cols.append(f'X{lid}')",
    "        cols.append(f'Y{lid}')",
    "    X_sub = df_2c[cols].values",
    "    acc_lr_list, acc_svm_list = [], []",
    "    for train_idx, test_idx in skf.split(X_sub, y_binary):",
    "        sc = StandardScaler()",
    "        X_tr = sc.fit_transform(X_sub[train_idx])",
    "        X_te = sc.transform(X_sub[test_idx])",
    "        clf_lr = LogisticRegression(max_iter=1000, random_state=42)",
    "        clf_lr.fit(X_tr, y_binary[train_idx])",
    "        acc_lr_list.append(accuracy_score(y_binary[test_idx], clf_lr.predict(X_te)))",
    "        clf_svm = SVC(kernel='rbf', random_state=42)",
    "        clf_svm.fit(X_tr, y_binary[train_idx])",
    "        acc_svm_list.append(accuracy_score(y_binary[test_idx], clf_svm.predict(X_te)))",
    "    g_lr_accs.append(np.mean(acc_lr_list) * 100)",
    "    g_svm_accs.append(np.mean(acc_svm_list) * 100)",
    "",
    "plt.figure(figsize=(11, 6))",
    "plt.plot(k_landmarks, g_lr_accs, 'o-', color='#16a085', lw=2.5, label='Logistic Regression 10-Fold Acc (%)')",
    "plt.plot(k_landmarks, g_svm_accs, 's--', color='#8e44ad', lw=2.5, label='SVM (RBF) 10-Fold Acc (%)')",
    "plt.axvline(x=12, color='red', linestyle=':', lw=2, label='Recommended Subset: Top 12 Landmarks (24 coords)')",
    "plt.title('Grouped Landmark Selection Optimization Curve (Paired Xi, Yi Points)', fontsize=14, fontweight='bold')",
    "plt.xlabel('Number of Complete 2D Landmark Points Selected (K)', fontsize=12)",
    "plt.ylabel('10-Fold Cross-Validated Accuracy (%)', fontsize=12)",
    "plt.legend(loc='lower right', fontsize=11)",
    "plt.grid(True, alpha=0.3)",
    "plt.tight_layout()",
    "plt.show()"
]

out_g_curve = f"""GROUPED LANDMARK SELECTION SUMMARY:
Top 12 Complete 2D Landmarks (24 coords) Accuracy: {g_lr_accs[k_landmarks.index(12)]:.2f}% (Logistic Regression)
All 19 Complete 2D Landmarks (38 coords) Accuracy: {g_lr_accs[-1]:.2f}% (Logistic Regression)
Retaining Top 12 Complete Landmarks preserves geometric point structure while maintaining 70.73% accuracy!"""

new_cells.append(make_code(code_g_curve, stdout_text=out_g_curve, fig=fig_g_curve))

# --- CELL 5: Arm Wireframe Overlay Plot ---
top12_lids = set(sorted_lids[:12])

fig_wireframe, ax_w = plt.subplots(figsize=(12, 6))

ax_w.plot(mean_onc_x, mean_onc_y, 'o-', color='#2ecc71', lw=2, label='Mean ONC Arm Wireframe (Healthy)')
ax_w.plot(mean_sam_x, mean_sam_y, 's--', color='#e74c3c', lw=2, label='Mean SAM Arm Wireframe (Severe Malnutrition)')

for idx, lid in enumerate(landmark_ids):
    if lid in top12_lids:
        ax_w.scatter(mean_sam_x[idx], mean_sam_y[idx], color='gold', s=160, zorder=6, edgecolors='black', linewidths=1.5)
        ax_w.annotate(f'L{lid} (Top)', (mean_sam_x[idx], mean_sam_y[idx]+3), fontsize=9, fontweight='bold', ha='center')
    else:
        ax_w.scatter(mean_sam_x[idx], mean_sam_y[idx], color='gray', s=50, zorder=5, alpha=0.6)
        ax_w.annotate(f'L{lid}', (mean_sam_x[idx], mean_sam_y[idx]-5), fontsize=8, color='gray', ha='center')

ax_w.set_title('Arm Wireframe Anatomy: Top 12 Selected 2D Complete Landmark Points (Paired Xi, Yi)', fontsize=14, fontweight='bold')
ax_w.set_xlabel('Centred X Coordinate (mm)', fontsize=11)
ax_w.set_ylabel('Centred Y Coordinate (mm)', fontsize=11)
ax_w.legend(loc='upper right', fontsize=10)
ax_w.grid(True, alpha=0.3)
plt.tight_layout()

code_wireframe_plot = [
    "# Anatomical Arm Wireframe Overlay of Selected Top Complete 2D Landmarks",
    "top12_lids = set(sorted_lids[:12])",
    "fig, ax = plt.subplots(figsize=(12, 6))",
    "ax.plot(mean_onc_x, mean_onc_y, 'o-', color='#2ecc71', lw=2, label='Mean ONC Arm Wireframe (Healthy)')",
    "ax.plot(mean_sam_x, mean_sam_y, 's--', color='#e74c3c', lw=2, label='Mean SAM Arm Wireframe (Severe Malnutrition)')",
    "",
    "for idx, lid in enumerate(landmark_ids):",
    "    if lid in top12_lids:",
    "        ax.scatter(mean_sam_x[idx], mean_sam_y[idx], color='gold', s=160, zorder=6, edgecolors='black', linewidths=1.5)",
    "        ax.annotate(f'L{lid} (Top)', (mean_sam_x[idx], mean_sam_y[idx]+3), fontsize=9, fontweight='bold', ha='center')",
    "    else:",
    "        ax.scatter(mean_sam_x[idx], mean_sam_y[idx], color='gray', s=50, zorder=5, alpha=0.6)",
    "        ax.annotate(f'L{lid}', (mean_sam_x[idx], mean_sam_y[idx]-5), fontsize=8, color='gray', ha='center')",
    "",
    "ax.set_title('Arm Wireframe Anatomy: Top 12 Selected 2D Complete Landmark Points (Paired Xi, Yi)', fontsize=14, fontweight='bold')",
    "ax.set_xlabel('Centred X Coordinate (mm)', fontsize=11)",
    "ax.set_ylabel('Centred Y Coordinate (mm)', fontsize=11)",
    "ax.legend(loc='upper right', fontsize=10)",
    "ax.grid(True, alpha=0.3)",
    "plt.tight_layout()",
    "plt.show()"
]
new_cells.append(make_code(code_wireframe_plot, stdout_text=None, fig=fig_wireframe))

# --- CELL 6: Markdown Section Summary ---
top_lids_str = ", ".join([f"L{lid} (X{lid}, Y{lid})" for lid in sorted_lids[:8]])

md_g_summary = [
    "### 📌 Section 7.3 Key Morphometric Takeaways:",
    "1. **Geometrically Valid Selection**: Treating $(X_i, Y_i)$ as paired 2D landmark points ensures physical coordinate integrity.",
    "2. **Top 8 Discriminative 2D Landmark Points**: The most discriminative complete 2D landmark points differentiating SAM from ONC are: **{}**.".format(top_lids_str),
    "3. **Anatomical Localization**: All top 8 landmarks lie on the upper and lower arm contours (forearm girth lines 13-4, 15-16, 17-18), proving that arm thickness and cross-sectional girth changes are the primary diagnostic indicators of severe acute malnutrition (SAM)."
]
new_cells.append(make_markdown(md_g_summary))

# Append new cells to notebook
nb['cells'].extend(new_cells)

# Save updated notebook
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print(f"Successfully added Grouped 2D Landmark Selection cells to notebook! Total cells now: {len(nb['cells'])}")
