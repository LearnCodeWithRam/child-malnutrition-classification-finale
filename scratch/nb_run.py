import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
from scipy.spatial.distance import cdist

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')

# Style configuration
np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

print("Environment successfully configured and dependencies loaded!", flush=True)
print(f"NumPy: {np.__version__} | Pandas: {pd.__version__}", flush=True)


# --- CELL ---

# Load dataset
df_path = "final_balanced_dataset_centred.csv"
try:
    df = pd.read_csv(df_path)
except FileNotFoundError:
    df = pd.read_csv("GM_out_of_sample_classification-ap-govt/final_balanced_dataset_centred.csv")

print("="*60, flush=True)
print("DATASET OVERVIEW", flush=True)
print("="*60, flush=True)
print(f"Dataset Shape: {df.shape}", flush=True)
print(f"Classes Present: {df['Class_Type'].unique()}", flush=True)
print("\nClass Distribution:", flush=True)
print(df['Class_Type'].value_counts(), flush=True)

# Extract X and Y landmark columns
x_cols = [c for c in df.columns if c.startswith('X')]
y_cols = [c for c in df.columns if c.startswith('Y')]

landmark_ids = [int(c[1:]) for c in x_cols]
k = len(landmark_ids)
n = len(df)

# Map Landmark ID to 0-based array index
id_to_idx = {lid: idx for idx, lid in enumerate(landmark_ids)}

# Define Arm Wireframe Contour Connections:
# 1. Upper Contour (Solid): 1 -> 12 -> 14 -> 16 -> 18 -> 10 -> 8 -> 6 -> 20
upper_contour_ids = [1, 12, 14, 16, 18, 10, 8, 6, 20]
upper_contour_idx = [id_to_idx[lid] for lid in upper_contour_ids]

# 2. Lower Contour (Solid): 4 -> 13 -> 15 -> 17 -> 9 -> 7 -> 5 -> 19
lower_contour_ids = [4, 13, 15, 17, 9, 7, 5, 19]
lower_contour_idx = [id_to_idx[lid] for lid in lower_contour_ids]

# 3. Outer Boundaries (Solid): Shoulder [1, 4], Wrist [20, 3, 19]
shoulder_edge_idx = [id_to_idx[1], id_to_idx[4]]
wrist_edge_idx = [id_to_idx[20], id_to_idx[3], id_to_idx[19]]

# 4. Transverse Cross-Sections (Dashed): 12-4, 14-13, 16-15, 18-17, 10-9, 8-7, 6-5
dashed_cross_ids = [
    [12, 4],
    [14, 13],
    [16, 15],
    [18, 17],
    [10, 9],
    [8, 7],
    [6, 5]
]
dashed_cross_idx = [[id_to_idx[l1], id_to_idx[l2]] for l1, l2 in dashed_cross_ids]

# 5. Midline Center Line (Dashed): 2 -> 3
midline_idx = [id_to_idx[2], id_to_idx[3]]

print(f"\nTotal Landmark Points (k): {k}", flush=True)
print(f"Landmark IDs: {landmark_ids}", flush=True)
print(f"Solid Boundary Contours defined: Upper (9 pts), Lower (8 pts), Wrist/Shoulder edges", flush=True)
print(f"Dashed Internal Cross-Sections defined: {len(dashed_cross_ids)} vertical links + Midline (2-3)", flush=True)

# Convert DataFrame into 3D NumPy shape array (k, 2, n)
M = np.zeros((k, 2, n))
for i in range(n):
    M[:, 0, i] = df.iloc[i][x_cols].values.astype(float)
    M[:, 1, i] = df.iloc[i][y_cols].values.astype(float)

true_class = df['Class_Type'].values
class_colors = {'sam': '#e74c3c', 'mam': '#f39c12', 'onc': '#2ecc71'}

print(f"\n3D Landmark Tensor constructed: {M.shape} (k={k}, coords=2, samples={n})", flush=True)


# --- CELL ---

# Helper Function: Plot Complete Arm Wireframe (Solid Contours + Dashed Cross-Sections)
def plot_arm_wireframe(ax, shapes, color, alpha_solid=0.8, alpha_dashed=0.5, lw_solid=1.8, lw_dashed=1.2, alpha_scatter=0.7, s=20, label=None):
    if shapes.ndim == 2:
        shapes = shapes[:, :, np.newaxis]
    
    n_samp = shapes.shape[2]
    
    # 1. Scatter landmark points
    all_x = shapes[:, 0, :].flatten()
    all_y = shapes[:, 1, :].flatten()
    ax.scatter(all_x, all_y, color=color, alpha=alpha_scatter, s=s, zorder=6)
    
    # 2. Build Solid Lines: Upper Contour, Lower Contour, Shoulder Edge, Wrist Edge
    solid_x, solid_y = [], []
    for i in range(n_samp):
        shp = shapes[:, :, i]
        # Upper Contour
        solid_x.extend(shp[upper_contour_idx, 0]); solid_x.append(np.nan)
        solid_y.extend(shp[upper_contour_idx, 1]); solid_y.append(np.nan)
        # Lower Contour
        solid_x.extend(shp[lower_contour_idx, 0]); solid_x.append(np.nan)
        solid_y.extend(shp[lower_contour_idx, 1]); solid_y.append(np.nan)
        # Shoulder Edge (1-4)
        solid_x.extend(shp[shoulder_edge_idx, 0]); solid_x.append(np.nan)
        solid_y.extend(shp[shoulder_edge_idx, 1]); solid_y.append(np.nan)
        # Wrist Edge (20-3-19)
        solid_x.extend(shp[wrist_edge_idx, 0]); solid_x.append(np.nan)
        solid_y.extend(shp[wrist_edge_idx, 1]); solid_y.append(np.nan)
        
    ax.plot(solid_x, solid_y, color=color, alpha=alpha_solid, linewidth=lw_solid, linestyle='-', label=label, zorder=5)
    
    # 3. Build Dashed Lines: Transverse Cross-Sections (12-4, 14-13, etc.) + Midline (2-3)
    dashed_x, dashed_y = [], []
    for i in range(n_samp):
        shp = shapes[:, :, i]
        for seg in dashed_cross_idx:
            dashed_x.extend(shp[seg, 0]); dashed_x.append(np.nan)
            dashed_y.extend(shp[seg, 1]); dashed_y.append(np.nan)
        # Midline (2-3)
        dashed_x.extend(shp[midline_idx, 0]); dashed_x.append(np.nan)
        dashed_y.extend(shp[midline_idx, 1]); dashed_y.append(np.nan)
        
    ax.plot(dashed_x, dashed_y, color=color, alpha=alpha_dashed, linewidth=lw_dashed, linestyle='--', zorder=4)

# Function 1: Center Shape (Remove Translation)
def center_shape(shape):
    centroid = np.mean(shape, axis=0)
    return shape - centroid, centroid

# Function 2: Scale Shape (Remove Scale)
def scale_shape(shape):
    centroid_size = np.sqrt(np.sum(shape ** 2))
    return shape / centroid_size, centroid_size

# Function 3: Ordinary Procrustes Analysis (OPA) (Remove Rotation via SVD)
def proc_opa(target, source):
    H = source.T @ target
    U, Sigma, Vt = svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
        
    aligned = source @ R
    angle = np.arctan2(R[1, 0], R[0, 0])
    return aligned, R, angle

# Function 4: Generalized Procrustes Analysis (GPA) with Mean or Functional Median Consensus
def proc_gpa(shapes, consensus_type='mean', max_iterations=20, tolerance=1e-6):
    k, m, n_samples = shapes.shape
    aligned = np.zeros_like(shapes)
    
    for i in range(n_samples):
        centered, _ = center_shape(shapes[:, :, i])
        scaled, _ = scale_shape(centered)
        aligned[:, :, i] = scaled
        
    mshape = aligned[:, :, 0].copy()
    for iteration in range(max_iterations):
        mshape_old = mshape.copy()
        for i in range(n_samples):
            aligned[:, :, i], _, _ = proc_opa(mshape, aligned[:, :, i])
            
        if consensus_type == 'median':
            mshape = np.median(aligned, axis=2)
        else:
            mshape = np.mean(aligned, axis=2)
            
        mshape, _ = scale_shape(mshape)
        if np.sum((mshape - mshape_old) ** 2) < tolerance:
            break
            
    return {'rotated': aligned, 'mshape': mshape, 'iterations': iteration + 1}

# Function 5: Calculate Centroid Sizes for Dataset
def get_centroid_sizes(shapes):
    n_samples = shapes.shape[2]
    sizes = np.zeros(n_samples)
    for i in range(n_samples):
        centered, _ = center_shape(shapes[:, :, i])
        _, cs = scale_shape(centered)
        sizes[i] = cs
    return sizes

# Function 6: Allometric Correction (Multivariate Regression of Shape Features on Centroid Size)
def apply_allometric_correction(X_train, CS_train, X_test=None, CS_test=None):
    reg = LinearRegression()
    CS_tr_arr = np.array(CS_train).reshape(-1, 1)
    reg.fit(CS_tr_arr, X_train)
    
    X_train_pred = reg.predict(CS_tr_arr)
    X_train_res = X_train - X_train_pred + np.mean(X_train, axis=0)
    
    if X_test is not None and CS_test is not None:
        CS_te_arr = np.array(CS_test).reshape(-1, 1)
        X_test_pred = reg.predict(CS_te_arr)
        X_test_res = X_test - X_test_pred + np.mean(X_train, axis=0)
        return X_train_res, X_test_res
        
    return X_train_res

# Function 7: Rotate Shapes Consistently for Visualization
def rotate_shapes(shapes, ms):
    n_samples = shapes.shape[2]
    alpha = ms[2, :] - ms[0, :]
    theta = np.arctan2(alpha[1], alpha[0])
    R = np.array([[np.cos(-theta), -np.sin(-theta)], 
                  [np.sin(-theta), np.cos(-theta)]])
    
    rot_shapes = np.zeros_like(shapes)
    for i in range(n_samples):
        rot_shapes[:, :, i] = shapes[:, :, i] @ R
    rot_ms = ms @ R
    return rot_shapes, rot_ms

# Function 8: Flatten Shapes to Feature Matrix
def shapes_to_features(shapes):
    k, m, n_samples = shapes.shape
    features = np.zeros((n_samples, k * m))
    for i in range(n_samples):
        features[i, :] = shapes[:, :, i].T.flatten()
    return features

# Function 9: Outlier Detection via Procrustes Distance
def detect_outliers_procrustes(shapes, mshape, threshold=2.5):
    n_samples = shapes.shape[2]
    distances = np.zeros(n_samples)
    for i in range(n_samples):
        distances[i] = np.sqrt(np.sum((shapes[:, :, i] - mshape) ** 2))
    
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    outlier_mask = distances > (mean_dist + threshold * std_dist)
    return distances, outlier_mask, mean_dist, std_dist

print("All Core Geometric Morphometrics & Arm Wireframe Plotting functions defined!", flush=True)


# --- CELL ---

# STEP 1: Raw Landmark Scatter Plot & Arm Wireframes (Before Alignment)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

classes = ['sam', 'mam', 'onc']
titles = ['SAM (Severe Acute Malnutrition)', 'MAM (Moderate Acute Malnutrition)', 'ONC (Optimal Nutritional Condition)']

for ax, cls, title in zip(axes, classes, titles):
    indices = np.where(true_class == cls)[0]
    plot_arm_wireframe(ax, M[:, :, indices], color=class_colors[cls], 
                       alpha_solid=0.15, alpha_dashed=0.08, lw_solid=0.9, lw_dashed=0.6, alpha_scatter=0.25, s=12)
    
    ax.set_title(f'Step 1: Raw Arm Wireframe Scatter Plot\n{title} (n={len(indices)})', fontsize=12, fontweight='bold')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

plt.suptitle('Step 1: Raw Arm Landmark Coordinates & Full Wireframe Grids (Before Alignment)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# --- CELL ---

# STEP 2: Centered Landmark Scatter Plot (Translation Removed)
M_centered = np.zeros_like(M)
for i in range(n):
    M_centered[:, :, i], _ = center_shape(M[:, :, i])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, cls, title in zip(axes, classes, titles):
    indices = np.where(true_class == cls)[0]
    plot_arm_wireframe(ax, M_centered[:, :, indices], color=class_colors[cls], 
                       alpha_solid=0.15, alpha_dashed=0.08, lw_solid=0.9, lw_dashed=0.6, alpha_scatter=0.25, s=12)
    
    ax.scatter(0, 0, color='black', s=80, marker='X', label='Centroid (0,0)', zorder=10)
    ax.set_title(f'Step 2: Centered Arm Wireframes\n{title}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Centered X coordinate')
    ax.set_ylabel('Centered Y coordinate')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

plt.suptitle('Step 2: Centered Landmark Coordinates & Wireframes (Translation Removed)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# --- CELL ---

# STEP 3: Scaled Landmark Scatter Plot (Centroid Size Normalized)
M_scaled = np.zeros_like(M_centered)
for i in range(n):
    M_scaled[:, :, i], _ = scale_shape(M_centered[:, :, i])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, cls, title in zip(axes, classes, titles):
    indices = np.where(true_class == cls)[0]
    plot_arm_wireframe(ax, M_scaled[:, :, indices], color=class_colors[cls], 
                       alpha_solid=0.15, alpha_dashed=0.08, lw_solid=0.9, lw_dashed=0.6, alpha_scatter=0.25, s=12)
    
    ax.set_title(f'Step 3: Scaled Arm Wireframes\n{title}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Scaled X coordinate')
    ax.set_ylabel('Scaled Y coordinate')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

plt.suptitle('Step 3: Centroid-Size Scaled Landmarks & Wireframes (Scale Removed)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# --- CELL ---

# STEP 4: Full GPA Aligned Landmark Scatter Plot (Translation, Scale, Rotation Removed)
gpa_all = proc_gpa(M, consensus_type='mean')
aligned_shapes = gpa_all['rotated']
mean_shape_overall = gpa_all['mshape']

# Consistently rotate for clean horizontal display
aligned_shapes_rot, mean_shape_rot = rotate_shapes(aligned_shapes, mean_shape_overall)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, cls, title in zip(axes, classes, titles):
    indices = np.where(true_class == cls)[0]
    plot_arm_wireframe(ax, aligned_shapes_rot[:, :, indices], color=class_colors[cls], 
                       alpha_solid=0.15, alpha_dashed=0.08, lw_solid=0.9, lw_dashed=0.6, alpha_scatter=0.25, s=12)
    
    # Overlay mean shape wireframe
    cls_mean = np.mean(aligned_shapes_rot[:, :, indices], axis=2)
    plot_arm_wireframe(ax, cls_mean, color='black', 
                       alpha_solid=1.0, alpha_dashed=0.8, lw_solid=2.5, lw_dashed=1.5, alpha_scatter=1.0, s=60, label='Class Mean Wireframe')
    
    ax.set_title(f'Step 4: GPA Aligned Wireframes\n{title}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Aligned X coordinate')
    ax.set_ylabel('Aligned Y coordinate')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

plt.suptitle('Step 4: Full Generalized Procrustes Analysis (GPA) Aligned Arm Wireframes', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# --- CELL ---

# STEP 5: Mean Shape Comparison Arm Wireframes (SAM vs MAM vs ONC)
sam_indices = np.where(true_class == 'sam')[0]
mam_indices = np.where(true_class == 'mam')[0]
onc_indices = np.where(true_class == 'onc')[0]

sam_mean = np.mean(aligned_shapes_rot[:, :, sam_indices], axis=2)
mam_mean = np.mean(aligned_shapes_rot[:, :, mam_indices], axis=2)
onc_mean = np.mean(aligned_shapes_rot[:, :, onc_indices], axis=2)

fig, ax = plt.subplots(figsize=(14, 8))

# Plot SAM Mean Shape Wireframe
plot_arm_wireframe(ax, sam_mean, color='#e74c3c', 
                   alpha_solid=1.0, alpha_dashed=0.8, lw_solid=2.5, lw_dashed=1.5, alpha_scatter=1.0, s=70, label='SAM Mean Wireframe')

# Plot MAM Mean Shape Wireframe
plot_arm_wireframe(ax, mam_mean, color='#f39c12', 
                   alpha_solid=1.0, alpha_dashed=0.8, lw_solid=2.5, lw_dashed=1.5, alpha_scatter=1.0, s=70, label='MAM Mean Wireframe')

# Plot ONC Mean Shape Wireframe
plot_arm_wireframe(ax, onc_mean, color='#2ecc71', 
                   alpha_solid=1.0, alpha_dashed=0.8, lw_solid=2.5, lw_dashed=1.5, alpha_scatter=1.0, s=70, label='ONC Mean Wireframe')

# Annotate landmark numbers on ONC mean shape
for i in range(k):
    ax.annotate(f"{landmark_ids[i]}", (onc_mean[i, 0], onc_mean[i, 1]), 
                textcoords="offset points", xytext=(0,7), ha='center', fontsize=10, fontweight='bold')

ax.set_title('Step 5: Morphological Comparison of Mean Arm Wireframes (SAM vs MAM vs ONC)\nSolid: Contours (1-12-14-16-18-10-8-6-20 & 4-13-15-17-9-7-5-19) | Dashed: Cross-Sections (12-4, 14-13, ...) & Midline (2-3)', fontsize=13, fontweight='bold')
ax.set_xlabel('Aligned X coordinate', fontsize=12)
ax.set_ylabel('Aligned Y coordinate', fontsize=12)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3)
ax.axis('equal')

plt.tight_layout()
plt.show()


# --- CELL ---

# STEP 6: Data Quality & Procrustes Distance Analysis
distances_all, outliers_all, mean_d, std_d = detect_outliers_procrustes(aligned_shapes, mean_shape_overall, threshold=2.5)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Bar chart of Procrustes Distances
colors_dist = ['#e74c3c' if is_out else '#3498db' for is_out in outliers_all]
axes[0].bar(range(n), distances_all, color=colors_dist, alpha=0.7)
axes[0].axhline(mean_d + 2.5 * std_d, color='red', linestyle='--', linewidth=2, label=f'Threshold (Mean + 2.5 SD = {mean_d+2.5*std_d:.4f})')
axes[0].axhline(mean_d, color='green', linestyle='-', linewidth=2, label=f'Mean Distance ({mean_d:.4f})')
axes[0].set_title('Step 6: Individual Procrustes Distances from Mean Shape', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Individual Index')
axes[0].set_ylabel('Procrustes Distance')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Subplot 2: Distance Distribution Boxplot by Class
df_dist = pd.DataFrame({'Class': true_class, 'Procrustes_Distance': distances_all})
sns.boxplot(x='Class', y='Procrustes_Distance', data=df_dist, palette=class_colors, ax=axes[1])
sns.stripplot(x='Class', y='Procrustes_Distance', data=df_dist, color='black', alpha=0.4, jitter=0.2, ax=axes[1])
axes[1].set_title('Procrustes Distance Distribution by Class', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Procrustes Distance')

plt.suptitle('Step 6: Data Quality & Outlier Detection Analysis', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print(f"Outlier Detection Summary:", flush=True)
print(f"   Mean Procrustes Distance: {mean_d:.4f} +- {std_d:.4f}", flush=True)
print(f"   Outliers Detected (> 2.5 SD): {np.sum(outliers_all)} / {n} ({np.sum(outliers_all)/n*100:.1f}%)", flush=True)


# --- CELL ---

# STEP 7: 2D PCA Shape Space Scatter Plot (PC1 vs PC2) with 95% Confidence Ellipses
X_features = shapes_to_features(aligned_shapes)

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_features)
var_exp = pca.explained_variance_ratio_ * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def draw_confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = np.mean(x), np.mean(y)
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

for cls in classes:
    idx = np.where(true_class == cls)[0]
    axes[0].scatter(X_pca[idx, 0], X_pca[idx, 1], label=cls.upper(), color=class_colors[cls], alpha=0.7, s=40)
    draw_confidence_ellipse(X_pca[idx, 0], X_pca[idx, 1], axes[0], n_std=2.0, 
                            edgecolor=class_colors[cls], linestyle='--', linewidth=2)

axes[0].set_title(f'Step 7: 2D PCA Shape Space Scatter Plot\nPC1 ({var_exp[0]:.1f}%) vs PC2 ({var_exp[1]:.1f}%)', fontsize=12, fontweight='bold')
axes[0].set_xlabel(f'PC1 ({var_exp[0]:.1f}% Variance)')
axes[0].set_ylabel(f'PC2 ({var_exp[1]:.1f}% Variance)')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Subplot 2: Scree Plot & Cumulative Variance
axes[1].bar(range(1, 11), var_exp[:10], color='#34495e', alpha=0.7, label='Individual Variance')
axes[1].plot(range(1, 11), np.cumsum(var_exp[:10]), 'ro-', linewidth=2, label='Cumulative Variance')
axes[1].axhline(90, color='green', linestyle='--', label='90% Variance Threshold')
axes[1].set_title('Scree Plot & Cumulative Variance Explained', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Principal Component')
axes[1].set_ylabel('Variance Explained (%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Step 7: PCA Shape Space Analysis & Variance Breakdown', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# --- CELL ---

# Helper Function to Compute Multi-Class Macro Sensitivity and Specificity
def calc_multiclass_metrics(y_true, y_pred, labels=['sam', 'mam', 'onc']):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n_cls = len(labels)
    
    acc = accuracy_score(y_true, y_pred)
    sens_list = []
    spec_list = []
    
    total = np.sum(cm)
    for c in range(n_cls):
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        fp = np.sum(cm[:, c]) - tp
        tn = total - (tp + fn + fp)
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        sens_list.append(sens)
        spec_list.append(spec)
        
    macro_sens = np.mean(sens_list)
    macro_spec = np.mean(spec_list)
    
    return acc, macro_sens, macro_spec

# Full Out-of-Sample LOOCV Protocol for Before & After Allometric Correction
def oos_loocv_allometric_experiment(shapes_tensor, labels, consensus_type='mean'):
    k, m, n_samples = shapes_tensor.shape
    
    # Pre-calculate raw Centroid Sizes
    centroid_sizes = get_centroid_sizes(shapes_tensor)
    
    classifiers = {
        'LDA': LinearDiscriminantAnalysis(),
        'LR': LogisticRegression(max_iter=1000, random_state=42),
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'RF': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
        'GB': GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    }
    
    preds_before = {name: [] for name in classifiers.keys()}
    preds_after = {name: [] for name in classifiers.keys()}
    
    print(f"Launching LOOCV Allometric Experiment ({consensus_type.upper()} Consensus, {n_samples} samples)...", flush=True)
    
    for i in range(n_samples):
        # 1. Leave individual i out
        train_shapes = np.delete(shapes_tensor, i, axis=2)
        train_labels = np.delete(labels, i)
        train_cs = np.delete(centroid_sizes, i)
        
        test_shape = shapes_tensor[:, :, i]
        test_cs = centroid_sizes[i]
        
        # 2. Run GPA on training set only (Mean or Median consensus)
        gpa_train = proc_gpa(train_shapes, consensus_type=consensus_type, max_iterations=10)
        X_train_raw = shapes_to_features(gpa_train['rotated'])
        ref_shape_train = gpa_train['mshape']
        
        # 3. Align test shape via fOPA onto ref_shape_train
        test_centered, _ = center_shape(test_shape)
        test_scaled, _ = scale_shape(test_centered)
        test_aligned, _, _ = proc_opa(ref_shape_train, test_scaled)
        X_test_raw = test_aligned.T.flatten().reshape(1, -1)
        
        # 4. Perform Allometric Correction (Residuals after CS Regression)
        X_train_allo, X_test_allo = apply_allometric_correction(X_train_raw, train_cs, X_test_raw, test_cs)
        
        # 5. Fit classifiers & predict
        for name, clf in classifiers.items():
            # Before Size Correction
            clf.fit(X_train_raw, train_labels)
            preds_before[name].append(clf.predict(X_test_raw)[0])
            
            # After Size Correction
            clf.fit(X_train_allo, train_labels)
            preds_after[name].append(clf.predict(X_test_allo)[0])
            
        if (i + 1) % 100 == 0 or (i + 1) == n_samples:
            print(f"   Completed {i + 1}/{n_samples} folds...", flush=True)
            
    # Compute Metrics Table
    records = []
    for name in classifiers.keys():
        acc_b, sens_b, spec_b = calc_multiclass_metrics(labels, preds_before[name])
        acc_a, sens_a, spec_a = calc_multiclass_metrics(labels, preds_after[name])
        
        records.append({
            'Model': name,
            'Consensus': f'fOPA to {consensus_type}',
            'Before_Acc': acc_b, 'Before_Sens': sens_b, 'Before_Spec': spec_b,
            'After_Acc': acc_a, 'After_Sens': sens_a, 'After_Spec': spec_a
        })
        
    return pd.DataFrame(records), preds_before, preds_after

print("Out-of-sample LOOCV allometric experiment module ready!", flush=True)


# --- CELL ---

# Run Allometric Experiments for both fOPA to Mean and fOPA to Median shape
df_res_mean, preds_bef_mean, preds_aft_mean = oos_loocv_allometric_experiment(M, true_class, consensus_type='mean')
df_res_median, preds_bef_median, preds_aft_median = oos_loocv_allometric_experiment(M, true_class, consensus_type='median')

print("\nOut-of-sample allometric cross-validation experiments complete!", flush=True)


# --- CELL ---

print("="*85, flush=True)
print("GEOMETRIC MORPHOMETRICS ALLOMETRIC EFFECT COMPARISON (SAM / MAM / ONC)")
print("="*85, flush=True)

def print_formatted_allometric_table(df_res, consensus_label):
    print(f"\nConfiguration: 19 Landmarks - 38 Variables ({consensus_label})")
    print("-" * 85)
    print(f"{'Method':<8} | {'Allometric Effect':<24} | {'Accuracy':<8} | {'Sensitivity':<11} | {'Specificity':<11}")
    print("-" * 85)
    
    for _, row in df_res.iterrows():
        model = row['Model']
        print(f"{model:<8} | {'Before size correction':<24} | {row['Before_Acc']:<8.4f} | {row['Before_Sens']:<11.4f} | {row['Before_Spec']:<11.4f}")
        print(f"{'':<8} | {'After size correction':<24} | {row['After_Acc']:<8.4f} | {row['After_Sens']:<11.4f} | {row['After_Spec']:<11.4f}")
        print("-" * 85)

print_formatted_allometric_table(df_res_mean, "fOPA to mean shape")
print_formatted_allometric_table(df_res_median, "fOPA to functional median shape")


# --- CELL ---

# Comparative Bar Chart of Before vs After Allometric Size Correction
plt.figure(figsize=(14, 6))

df_plot = []
for _, row in df_res_mean.iterrows():
    df_plot.append({'Model': row['Model'], 'Allometric State': 'Before Size Correction', 'Accuracy (%)': row['Before_Acc']*100})
    df_plot.append({'Model': row['Model'], 'Allometric State': 'After Size Correction', 'Accuracy (%)': row['After_Acc']*100})

df_plot_df = pd.DataFrame(df_plot)

ax = sns.barplot(x='Model', y='Accuracy (%)', hue='Allometric State', data=df_plot_df, palette=['#e74c3c', '#2ecc71'])
plt.axhline(33.33, color='gray', linestyle='--', label='Chance Level (33.3%)')

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.1f}%',
                    (p.get_x() + p.get_width() / 2., height / 2.),
                    ha='center', va='center',
                    color='white', fontweight='bold', fontsize=10)

plt.title('Out-of-Sample Classification Accuracy: Before vs After Allometric Size Correction', fontsize=14, fontweight='bold')
plt.xlabel('Machine Learning Model', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(0, 100)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# --- CELL ---

# Grid of 3-class Confusion Matrices (After Allometric Correction - fOPA to Mean)
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

class_labels = ['sam', 'mam', 'onc']

for ax, (name, preds) in zip(axes, preds_aft_mean.items()):
    cm = confusion_matrix(true_class, preds, labels=class_labels)
    acc = accuracy_score(true_class, preds)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax, cbar=False,
                xticklabels=['SAM', 'MAM', 'ONC'], yticklabels=['SAM', 'MAM', 'ONC'],
                annot_kws={"size": 14, "weight": "bold"})
    
    ax.set_title(f'{name} (After Size Correction)\nAccuracy: {acc*100:.2f}%', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)

plt.suptitle('Multi-Class Confusion Matrices After Allometric Size Correction (LOOCV)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
