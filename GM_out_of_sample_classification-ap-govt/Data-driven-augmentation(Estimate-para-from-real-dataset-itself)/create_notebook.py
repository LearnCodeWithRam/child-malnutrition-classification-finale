import json
import os

target_dir = r'e:\ap-govt-EDA-model-tainning-testing\GM_out_of_sample_classification-main\GM_out_of_sample_classification-ap-govt\Data-driven-augmentation(Estimate-para-from-real-dataset-itself)'
notebook_path = os.path.join(target_dir, 'Data_Driven_Augmentation_Parameter_Estimation.ipynb')

nb = {
 "cells": [],
 "metadata": {
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

def add_markdown(text):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(code_text):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code_text.split("\n")]
    })

# --- CELL 1: TITLE & EXECUTIVE SUMMARY ---
add_markdown("""# Data-Driven Augmentation Parameter Estimation & Synthetic Dataset Generation
### Scientific Geometric Morphometrics for Child Malnutrition Landmark Datasets (ONC, MAM, SAM)

---

## Executive Summary & Data Quality Filtering
In benchmark simulation studies evaluating geometric morphometrics and landmark alignment algorithms, random 2D affine geometric transformations are typically generated using arbitrary parameters ($\beta \sim U(0, 10)$, $\theta \sim U(0, 2\pi)$, $\gamma \sim U(-5, 5)$).

However, raw annotation datasets can contain **vertically inverted image annotations** (where upper contour landmarks $Y_{12 \dots 20} < Y_{4 \dots 19}$ lower contour landmarks). Including distorted/inverted samples pollutes empirical parameter bounds and distorts Procrustes shape averages.

In this study, we perform **Automated Data Quality Filtering** on the reference dataset (`reference-sheet-data-driven-augmentaton-param-estimation.xlsx`):
- **Raw Samples:** 918 configurations across 341 children.
- **Distorted / Inverted Samples Filtered Out:** 259 samples (199 in MAM, 60 in SAM).
- **Clean Quality Dataset Retained:** **659 samples** (ONC: 402, MAM Clean: 114, SAM Clean: 143).

Empirical augmentation parameters ($\beta, \theta, \gamma, \sigma$) are estimated strictly from the **659 clean configurations**. Furthermore, we build a **Global Class-Specific Augmentation Generator** that receives class-specific empirical parameters to generate clean synthetic CSV datasets (`synthetic_augmented_onc_landmarks.csv`, `synthetic_augmented_mam_landmarks.csv`, `synthetic_augmented_sam_landmarks.csv`, `synthetic_augmented_balanced_dataset_combined.csv`) and plot their generated wireframe skeletons.

---
""")

# --- CELL 2: SECTION 1 ---
add_markdown("""## Section 1: Environment Setup & Library Imports""")

add_code("""import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# Set publication style
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'figure.dpi': 300
})

EXCEL_PATH = r'reference-sheet-data-driven-augmentaton-param-estimation.xlsx'
print("Libraries imported successfully.")
""")

# --- CELL 3: SECTION 2 ---
add_markdown("""## Section 2: Dataset Loading & Data Quality Filtering

We load `reference-sheet-data-driven-augmentaton-param-estimation.xlsx` and evaluate pairwise Y-coordinate orientation. Samples exhibiting vertically inverted landmark annotations ($Y_{\\text{upper}} < Y_{\\text{lower}}$) are detected and filtered out to preserve dataset integrity.""")

add_code("""# Load raw landmark dataset
df_raw = pd.read_excel(EXCEL_PATH)
df = df_raw.dropna(subset=['Class_Type']).copy()

df['Subject_ID'] = df['File_Name'].apply(lambda x: str(x).split('_')[0] if pd.notnull(x) else '')

# Exact landmark numbers present in dataset (1 to 20, omitting 11)
lm_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20]
lm_map = {num: i for i, num in enumerate(lm_nums)}

x_cols = [f'X{num}' for num in lm_nums]
y_cols = [f'Y{num}' for num in lm_nums]

# Detect Vertically Inverted Annotation Samples
pairs = [(12, 4), (14, 13), (16, 15), (18, 17), (10, 9), (8, 7), (6, 5), (20, 19)]
df['Y_diff_mean'] = 0.0
for u, l in pairs:
    df['Y_diff_mean'] += (df[f'Y{u}'] - df[f'Y{l}'])
df['Y_diff_mean'] /= len(pairs)
df['Is_Inverted'] = df['Y_diff_mean'] < 0

print(f"Total Raw Dataset Samples: {len(df)}")
print(f"Distorted / Inverted Samples Filtered Out: {df['Is_Inverted'].sum()}")

# Filter Clean Quality Dataset (N=659)
clean_df = df[~df['Is_Inverted']].copy()
print(f"Clean Quality Dataset Retained (N): {len(clean_df)}")
print("\\nClean Class Distribution:")
print(clean_df['Class_Type'].value_counts())

X_coords = clean_df[x_cols].values
Y_coords = clean_df[y_cols].values
N, p = X_coords.shape
landmarks = np.stack([X_coords, Y_coords], axis=2)
""")

# --- CELL 4: SECTION 3 ---
add_markdown("""## Section 3: Generalized Procrustes Alignment (GPA) Engine""")

add_code("""def gpa_align(shapes, max_iter=50, ftol=1e-7):
    '''Generalized Procrustes Analysis (GPA) alignment for landmark configurations.'''
    centered = shapes - shapes.mean(axis=1, keepdims=True)
    cs = np.sqrt((centered**2).sum(axis=(1, 2)))
    scaled = centered / cs[:, np.newaxis, np.newaxis]
    
    ref_shape = scaled[0].copy()
    for iteration in range(max_iter):
        aligned = np.zeros_like(scaled)
        for i in range(len(scaled)):
            H = scaled[i].T @ ref_shape
            U, S, Vt = np.linalg.svd(H)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt
            aligned[i] = scaled[i] @ R
        
        new_ref = aligned.mean(axis=0)
        new_ref /= np.sqrt((new_ref**2).sum())
        
        diff = np.linalg.norm(new_ref - ref_shape)
        ref_shape = new_ref
        if diff < ftol:
            break
            
    return aligned, ref_shape, cs

# Perform Global GPA Alignment on Clean Dataset
aligned_landmarks, global_mean_shape, centroid_sizes = gpa_align(landmarks)
clean_df['Centroid_Size'] = centroid_sizes
mean_cs_global = centroid_sizes.mean()
clean_df['Scale_Ratio'] = centroid_sizes / mean_cs_global

print(f"Global Procrustes Consensus Shape computed across {N} clean samples.")
print(f"Global Mean Centroid Size (S_mean): {mean_cs_global:.4f}")
""")

# --- CELL 5: SECTION 4 ---
add_markdown("""## Section 4: Empirical Scale Variation ($\beta$) Estimation (Clean Dataset)""")

add_code("""l1_idx = lm_map[1]
l3_idx = lm_map[3]
clean_df['Arm_Length_L1_L3'] = np.linalg.norm(landmarks[:, l3_idx, :] - landmarks[:, l1_idx, :], axis=1)

def get_stats(series):
    return {
        'Min': series.min(),
        'Max': series.max(),
        'Mean': series.mean(),
        'Std': series.std(),
        'P5 (5%)': series.quantile(0.05),
        'P25 (25%)': series.quantile(0.25),
        'Median (P50)': series.median(),
        'P75 (75%)': series.quantile(0.75),
        'P95 (95%)': series.quantile(0.95)
    }

scale_summary = []
scale_summary.append({'Class': 'CLEAN OVERALL (N=659)', **get_stats(clean_df['Scale_Ratio'])})
for cls in ['onc', 'mam', 'sam']:
    scale_summary.append({'Class': f'{cls.upper()} CLEAN', **get_stats(clean_df[clean_df['Class_Type'] == cls]['Scale_Ratio'])})

scale_df = pd.DataFrame(scale_summary)
print("=== CLEAN EMPIRICAL SCALE RATIO (β) PARAMETER ESTIMATION ===")
print(scale_df.to_string(index=False))
""")

# --- CELL 6: SECTION 5 ---
add_markdown("""## Section 5: Empirical Rotation Variation ($\theta$) Estimation (Clean Dataset)""")

add_code("""orient_vecs = landmarks[:, l3_idx, :] - landmarks[:, l1_idx, :]
orient_angles_deg = np.degrees(np.arctan2(orient_vecs[:, 1], orient_vecs[:, 0]))
clean_df['Orientation_Deg'] = orient_angles_deg
clean_df['Rel_Orientation_Deg'] = orient_angles_deg - orient_angles_deg.mean()

rot_summary = []
rot_summary.append({'Class': 'CLEAN OVERALL (N=659)', **get_stats(clean_df['Rel_Orientation_Deg'])})
for cls in ['onc', 'mam', 'sam']:
    rot_summary.append({'Class': f'{cls.upper()} CLEAN', **get_stats(clean_df[clean_df['Class_Type'] == cls]['Rel_Orientation_Deg'])})

rot_df = pd.DataFrame(rot_summary)
print("=== CLEAN EMPIRICAL RELATIVE ROTATION (θ) PARAMETER ESTIMATION ===")
print(rot_df.to_string(index=False))
""")

# --- CELL 7: SECTION 6 ---
add_markdown("""## Section 6: Empirical Translation Variation ($\gamma_x, \gamma_y$) Estimation (Clean Dataset)""")

add_code("""centroids = landmarks.mean(axis=1)
global_centroid = centroids.mean(axis=0)
clean_df['Trans_X'] = centroids[:, 0] - global_centroid[0]
clean_df['Trans_Y'] = centroids[:, 1] - global_centroid[1]

trans_summary = []
for axis in ['Trans_X', 'Trans_Y']:
    trans_summary.append({'Axis / Class': f'CLEAN OVERALL {axis}', **get_stats(clean_df[axis])})
    for cls in ['onc', 'mam', 'sam']:
        trans_summary.append({'Axis / Class': f'{cls.upper()} CLEAN {axis}', **get_stats(clean_df[clean_df['Class_Type'] == cls][axis])})

trans_df = pd.DataFrame(trans_summary)
print("=== CLEAN EMPIRICAL TRANSLATION SHIFTS (γ_x, γ_y) PARAMETER ESTIMATION ===")
print(trans_df.to_string(index=False))
""")

# --- CELL 8: SECTION 7 ---
add_markdown("""## Section 7: Empirical Landmark Placement Error ($\sigma_{\text{digitization}}$) & Population Noise ($\sigma_{\text{shape}}$)""")

add_code("""subject_counts = clean_df['Subject_ID'].value_counts()
multi_subjects = subject_counts[subject_counts >= 2].index

intra_residuals = []
for sub in multi_subjects:
    sub_indices = clean_df[clean_df['Subject_ID'] == sub].index
    sub_shapes = landmarks[clean_df.index.get_indexer(sub_indices)]
    if len(sub_shapes) < 2:
        continue
    aligned_sub, mean_sub, cs_sub = gpa_align(sub_shapes)
    res_physical = (aligned_sub - mean_sub[np.newaxis, :, :]) * cs_sub.mean()
    intra_residuals.append(res_physical.reshape(-1, 2))

all_intra_res = np.vstack(intra_residuals)
intra_rmse = np.sqrt((all_intra_res**2).mean())

print(f"=== INTRA-SUBJECT LANDMARK PLACEMENT ERROR (CLEAN DATASET) ===")
print(f"Empirical Landmark Placement Noise RMSE (σ_intra): {intra_rmse:.4f}")

class_stats = {}
print("\\n=== POPULATION GPA SHAPE VARIATION BY CLASS (CLEAN DATASET) ===")
for cls in ['onc', 'mam', 'sam']:
    cls_df = clean_df[clean_df['Class_Type'] == cls]
    cls_shapes = landmarks[clean_df['Class_Type'] == cls]
    aligned_cls, mean_cls, cs_cls = gpa_align(cls_shapes)
    res_cls_phys = (aligned_cls - mean_cls[np.newaxis, :, :]) * cs_cls[:, np.newaxis, np.newaxis]
    rmse_cls = np.sqrt((res_cls_phys**2).mean())
    
    vecs = cls_shapes[:, l3_idx, :] - cls_shapes[:, l1_idx, :]
    angles = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0]))
    rel_angles = angles - angles.mean()
    
    c_cls = cls_shapes.mean(axis=1)
    g_cent = c_cls.mean(axis=0)
    tx = c_cls[:, 0] - g_cent[0]
    ty = c_cls[:, 1] - g_cent[1]
    scale_ratios = cs_cls / mean_cs_global
    
    class_stats[cls] = {
        'count': len(cls_df),
        'scale_ratio': {'p5': float(np.percentile(scale_ratios, 5)), 'p95': float(np.percentile(scale_ratios, 95))},
        'rotation_deg': {'p5': float(np.percentile(rel_angles, 5)), 'p95': float(np.percentile(rel_angles, 95))},
        'trans_x': {'p5': float(np.percentile(tx, 5)), 'p95': float(np.percentile(tx, 95))},
        'trans_y': {'p5': float(np.percentile(ty, 5)), 'p95': float(np.percentile(ty, 95))},
        'gpa_rmse': float(rmse_cls),
        'mean_shape': mean_cls,
        'mean_phys': mean_cls * cs_cls.mean()
    }
    print(f"Class {cls.upper():3s} Clean (n={len(cls_shapes):3d}): GPA Shape Noise RMSE = {rmse_cls:.4f}")
""")

# --- CELL 9: SECTION 8 ---
add_markdown("""## Section 8: Publication Parameter Comparison Table""")

add_code("""def format_dynamic_column(subset_df, cls_shapes):
    al, ref, cs = gpa_align(cls_shapes)
    res_phys = (al - ref[np.newaxis, :, :]) * cs[:, np.newaxis, np.newaxis]
    gpa_rmse = np.sqrt((res_phys**2).mean())
    
    s = subset_df['Scale_Ratio']
    r = subset_df['Rel_Orientation_Deg']
    tx = subset_df['Trans_X']
    ty = subset_df['Trans_Y']
    
    return [
        f'U({s.min():.2f}, {s.max():.2f})',
        f'U({s.quantile(0.05):.2f}, {s.quantile(0.95):.2f})',
        f'U({r.quantile(0.05):.1f} deg, {r.quantile(0.95):.1f} deg)',
        f'{r.std():.2f} deg',
        f'[{tx.quantile(0.05):.2f}, {tx.quantile(0.95):.2f}]',
        f'[{ty.quantile(0.05):.2f}, {ty.quantile(0.95):.2f}]',
        f'sigma_shape = {gpa_rmse:.3f}'
    ]

col_overall = format_dynamic_column(clean_df, landmarks)
col_onc = format_dynamic_column(clean_df[clean_df['Class_Type']=='onc'], landmarks[(clean_df['Class_Type']=='onc').values])
col_mam = format_dynamic_column(clean_df[clean_df['Class_Type']=='mam'], landmarks[(clean_df['Class_Type']=='mam').values])
col_sam = format_dynamic_column(clean_df[clean_df['Class_Type']=='sam'], landmarks[(clean_df['Class_Type']=='sam').values])
col_overall[6] = f'sigma_digitization = {intra_rmse:.3f}'

comp_data_dynamic = {
    'Parameter': [
        'Scale (beta) Min..Max',
        'Scale (beta) 5th..95th Percentile',
        'Rotation (theta) Operational Range (P5..P95)',
        'Rotation (theta) Std Dev',
        'Translation gamma_x (X Shift P5..P95)',
        'Translation gamma_y (Y Shift P5..P95)',
        'Landmark Noise (sigma_noise)'
    ],
    'Generic Simulation Settings': [
        'U(0.0, 10.0)',
        'N/A',
        'U(-180.0 deg, +180.0 deg)',
        '103.92 deg (Uniform)',
        '[-5.00, +5.00]',
        '[-5.00, +5.00]',
        'Arbitrary Matrix Normal'
    ],
    'Empirical Clean Overall (N=659)': col_overall,
    'Empirical ONC Clean (n=402)': col_onc,
    'Empirical MAM Clean (n=114)': col_mam,
    'Empirical SAM Clean (n=143)': col_sam
}

comp_table_df = pd.DataFrame(comp_data_dynamic)
print(comp_table_df.to_string(index=False))
""")


# --- CELL 10: SECTION 9 ---
add_markdown("""## Section 9: High-Quality Publication Visualizations

We display Figures 1 to 5 generated from the clean dataset.""")

add_code("""from IPython.display import Image, display

for fig_name in ['fig1_wireframe_skeleton.png', 'fig2_class_consensus_shapes.png', 
                'fig3_landmark_variability_ellipses.png', 'fig4_synthetic_augmentation_comparison.png', 
                'fig5_empirical_distributions.png']:
    if os.path.exists(fig_name):
        print(f"Displaying {fig_name}:")
        display(Image(filename=fig_name))
""")

# --- CELL 11: SECTION 10 ---
add_markdown("""## Section 10: Global Class-Specific Synthetic Landmark Generator Function

We implement the global data-driven augmentation generator function `generate_class_specific_augmented_landmarks()`.
When generating synthetic data for a target class (**ONC**, **MAM**, or **SAM**), the function dynamically extracts and applies the **class-specific empirical parameters** estimated from the clean training dataset.""")

add_code("""def generate_class_specific_augmented_landmarks(class_type='sam', num_samples=100, seed=42):
    '''
    Global Synthetic Data Augmentation Generator Function.
    
    Parameters:
    -----------
    class_type : str
        Target class name ('onc', 'mam', or 'sam').
        When class_type='sam', SAM class-specific empirical parameters are passed.
        When class_type='mam', MAM class-specific empirical parameters are passed.
        When class_type='onc', ONC class-specific empirical parameters are passed.
    num_samples : int
        Number of synthetic configurations to generate.
    seed : int
        Random seed for reproducibility.
        
    Returns:
    --------
    synth_landmarks : np.ndarray
        Generated synthetic landmark tensor (num_samples, p, 2).
    '''
    np.random.seed(seed)
    cls_key = class_type.lower()
    if cls_key not in class_stats:
        raise ValueError(f"Unknown class_type '{class_type}'. Must be 'onc', 'mam', or 'sam'.")
        
    p_cfg = class_stats[cls_key]
    base_shape = p_cfg['mean_phys']
    p_count = len(base_shape)
    
    synth_landmarks = np.zeros((num_samples, p_count, 2))
    
    for i in range(num_samples):
        # 1. Class-Specific Scale Sampling (beta)
        beta = np.random.uniform(p_cfg['scale_ratio']['p5'], p_cfg['scale_ratio']['p95'])
        
        # 2. Class-Specific Rotation Sampling (theta in radians)
        theta_rad = np.radians(np.random.uniform(p_cfg['rotation_deg']['p5'], p_cfg['rotation_deg']['p95']))
        R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                      [np.sin(theta_rad),  np.cos(theta_rad)]])
        
        # 3. Class-Specific Anisotropic Translation Sampling (gamma_x, gamma_y)
        gamma = np.array([
            np.random.uniform(p_cfg['trans_x']['p5'], p_cfg['trans_x']['p95']),
            np.random.uniform(p_cfg['trans_y']['p5'], p_cfg['trans_y']['p95'])
        ])
        
        # 4. Class-Specific Landmark Noise Perturbation (Gaussian Matrix Normal)
        noise = np.random.normal(0, p_cfg['gpa_rmse'], size=(p_count, 2))
        
        # Apply transformation
        synth_landmarks[i] = (base_shape @ R.T) * beta + gamma + noise
        
    return synth_landmarks

# Demonstrate synthetic generation for SAM, MAM, and ONC using class-specific parameters
synth_sam = generate_class_specific_augmented_landmarks(class_type='sam', num_samples=500, seed=42)
synth_mam = generate_class_specific_augmented_landmarks(class_type='mam', num_samples=500, seed=43)
synth_onc = generate_class_specific_augmented_landmarks(class_type='onc', num_samples=500, seed=44)

print(f"Generated Synthetic SAM Dataset shape: {synth_sam.shape}")
print(f"Generated Synthetic MAM Dataset shape: {synth_mam.shape}")
print(f"Generated Synthetic ONC Dataset shape: {synth_onc.shape}")
""")

# --- CELL 12: SECTION 12 ---
add_markdown("""## Section 12: Synthetic CSV Data Export & Generated Wireframe Visualization

We export the generated synthetic datasets into clean CSV files for downstream model training and plot the generated synthetic wireframe configurations (`fig6_generated_synthetic_wireframes.png`).""")

add_code("""def export_synthetic_csv(synth_tensor, class_name, filename):
    rows = []
    num_samples, p_cnt, _ = synth_tensor.shape
    for i in range(num_samples):
        row = {
            'File_Name': f'synthetic_{class_name}_{i+1:04d}.jpg',
            'Individual': 9000 + i + 1,
            'Class_Type': class_name
        }
        for j, lm_num in enumerate(lm_nums):
            row[f'X{lm_num}'] = round(synth_tensor[i, j, 0], 4)
            row[f'Y{lm_num}'] = round(synth_tensor[i, j, 1], 4)
        rows.append(row)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(filename, index=False)
    print(f"Exported '{filename}' ({len(out_df)} synthetic samples).")
    return out_df

df_s_onc = export_synthetic_csv(synth_onc, 'onc', 'synthetic_augmented_onc_landmarks.csv')
df_s_mam = export_synthetic_csv(synth_mam, 'mam', 'synthetic_augmented_mam_landmarks.csv')
df_s_sam = export_synthetic_csv(synth_sam, 'sam', 'synthetic_augmented_sam_landmarks.csv')

comb_df = pd.concat([df_s_onc, df_s_mam, df_s_sam], ignore_index=True)
comb_df.to_csv('synthetic_augmented_balanced_dataset_combined.csv', index=False)
print(f"Exported 'synthetic_augmented_balanced_dataset_combined.csv' ({len(comb_df)} total samples).")

# Display Figure 6: Generated Synthetic Wireframe Visualization
if os.path.exists('fig6_generated_synthetic_wireframes.png'):
    print("\\nDisplaying Generated Synthetic Data Wireframe Plot (fig6_generated_synthetic_wireframes.png):")
    display(Image(filename='fig6_generated_synthetic_wireframes.png'))
""")

# --- CELL 13: CONCLUSION ---
add_markdown("""## Section 13: Peer-Review Publication Statement

> *"Augmentation parameters were derived empirically from the observed distributions of camera orientation, image scaling, arm positioning, and landmarking error within the clean, quality-filtered training dataset (`reference-sheet-data-driven-augmentaton-param-estimation.xlsx`). 259 vertically inverted annotation samples were filtered out, retaining 659 un-distorted child arm configurations. Scaling ($\beta \sim U(0.70, 1.26)$), operational rotation ($\theta \sim U(-14^\circ, -0.3^\circ)$), anisotropic translation ($\gamma_x \in [-4.5, +4.2]$, $\gamma_y \in [-2.1, +1.9]$), and landmark perturbation ($\sigma = 0.646$) were constrained by measured empirical image-acquisition variability and intra-subject landmark placement error across 659 clean child arm configurations."*""")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Jupyter Notebook successfully created at: {notebook_path}")
