import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# Target directory definition
TARGET_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(TARGET_DIR, 'final_cleaned_dataset_full.csv')

print("Starting Fully Dynamic Data-Driven Parameter Estimation & Visualizations...")

# 1. Load Data
df = pd.read_csv(CSV_PATH).dropna(subset=['Class_Type']).copy()
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

# Filter Clean Quality Dataset (N=659)
clean_df = df[~df['Is_Inverted']].copy()

X_coords = clean_df[x_cols].values
Y_coords = clean_df[y_cols].values
N, p = X_coords.shape
landmarks = np.stack([X_coords, Y_coords], axis=2) # (N, 19, 2)

# Pairwise Wireframe Structure Definitions:
MAIN_LM_NUMS = [1, 2, 3, 4]
MAIN_LM_NAMES = {1: 'Shoulder (L1)', 2: 'Elbow (L2)', 3: 'Wrist (L3)', 4: 'Armpit (L4)'}

upper_sequence = [1, 12, 14, 16, 18, 10, 8, 6, 20]
lower_sequence = [4, 13, 15, 17, 9, 7, 5, 19]
vertical_pairs = [(12, 4), (14, 13), (16, 15), (18, 17), (10, 9), (8, 7), (6, 5), (20, 19)]

# 2. Generalized Procrustes Analysis (GPA) Implementation
def gpa_align(shapes, max_iter=50, ftol=1e-7):
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

aligned_landmarks, mean_shape, centroid_sizes = gpa_align(landmarks)
clean_df['Centroid_Size'] = centroid_sizes
mean_cs_global = centroid_sizes.mean()
clean_df['Scale_Ratio'] = centroid_sizes / mean_cs_global

# 3. Orientation Calculation
l1_idx = lm_map[1]
l3_idx = lm_map[3]
orient_vecs = landmarks[:, l3_idx, :] - landmarks[:, l1_idx, :]
orient_angles_deg = np.degrees(np.arctan2(orient_vecs[:, 1], orient_vecs[:, 0]))
clean_df['Orientation_Deg'] = orient_angles_deg
mean_orient = orient_angles_deg.mean()
clean_df['Rel_Orientation_Deg'] = orient_angles_deg - mean_orient

# 4. Translation Calculation
centroids = landmarks.mean(axis=1)
global_centroid = centroids.mean(axis=0)
clean_df['Trans_X'] = centroids[:, 0] - global_centroid[0]
clean_df['Trans_Y'] = centroids[:, 1] - global_centroid[1]

# 5. Intra-Subject Placement Error
subject_counts = clean_df['Subject_ID'].value_counts()
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
intra_std_x = np.std(all_intra_res[:, 0])
intra_std_y = np.std(all_intra_res[:, 1])

# 6. Global & Class-Wise Population Shape Noise & Parameters
global_res_phys = (aligned_landmarks - mean_shape[np.newaxis, :, :]) * centroid_sizes[:, np.newaxis, np.newaxis]
global_rmse = np.sqrt((global_res_phys**2).mean())

class_stats = {}
for cls in ['onc', 'mam', 'sam']:
    cls_df = clean_df[clean_df['Class_Type'] == cls]
    cls_shapes = landmarks[(clean_df['Class_Type'] == cls).values]
    aligned_cls, mean_cls, cs_cls = gpa_align(cls_shapes)
    res_cls_phys = (aligned_cls - mean_cls[np.newaxis, :, :]) * cs_cls[:, np.newaxis, np.newaxis]
    
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
        'centroid_size': {
            'min': float(cls_df['Centroid_Size'].min()), 'max': float(cls_df['Centroid_Size'].max()),
            'mean': float(cls_df['Centroid_Size'].mean()), 'std': float(cls_df['Centroid_Size'].std()),
            'p5': float(cls_df['Centroid_Size'].quantile(0.05)), 'p95': float(cls_df['Centroid_Size'].quantile(0.95))
        },
        'scale_ratio': {
            'min': float(cls_df['Scale_Ratio'].min()), 'max': float(cls_df['Scale_Ratio'].max()),
            'mean': float(cls_df['Scale_Ratio'].mean()), 'std': float(cls_df['Scale_Ratio'].std()),
            'p5': float(np.percentile(scale_ratios, 5)), 'p95': float(np.percentile(scale_ratios, 95))
        },
        'rotation_deg': {
            'min': float(cls_df['Orientation_Deg'].min()), 'max': float(cls_df['Orientation_Deg'].max()),
            'mean': float(cls_df['Orientation_Deg'].mean()), 'std': float(cls_df['Orientation_Deg'].std()),
            'p5': float(np.percentile(rel_angles, 5)), 'p95': float(np.percentile(rel_angles, 95))
        },
        'trans_x': {
            'min': float(cls_df['Trans_X'].min()), 'max': float(cls_df['Trans_X'].max()),
            'mean': float(cls_df['Trans_X'].mean()), 'std': float(cls_df['Trans_X'].std()),
            'p5': float(np.percentile(tx, 5)), 'p95': float(np.percentile(tx, 95))
        },
        'trans_y': {
            'min': float(cls_df['Trans_Y'].min()), 'max': float(cls_df['Trans_Y'].max()),
            'mean': float(cls_df['Trans_Y'].mean()), 'std': float(cls_df['Trans_Y'].std()),
            'p5': float(np.percentile(ty, 5)), 'p95': float(np.percentile(ty, 95))
        },
        'gpa_rmse': float(np.sqrt((res_cls_phys**2).mean())),
        'std_x': float(np.std(res_cls_phys[:, :, 0])),
        'std_y': float(np.std(res_cls_phys[:, :, 1])),
        'mean_shape': mean_cls,
        'mean_phys': mean_cls * cs_cls.mean(),
        'aligned_shapes': aligned_cls,
        'centroid_sizes': cs_cls,
        'res_phys': res_cls_phys
    }

# 7. DYNAMICALLY CONSTRUCT COMPARISON TABLE DIRECTLY FROM COMPUTED VARIABLES
def format_dynamic_column(subset_df, cls_shapes):
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

df_comp_table = pd.DataFrame(comp_data_dynamic)
print("\n=== DYNAMICALLY COMPUTED PARAMETER COMPARISON TABLE ===")
print(df_comp_table.to_string(index=False))

# High-Quality Publication Aesthetics Setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 15,
    'legend.fontsize': 10,
    'figure.dpi': 300
})

colors = {'onc': '#27ae60', 'mam': '#f39c12', 'sam': '#c0392b'}
labels = {
    'onc': f'ONC (Optimal Nutrition, n={class_stats["onc"]["count"]})',
    'mam': f'MAM (Moderate Acute Malnutrition, n={class_stats["mam"]["count"]})',
    'sam': f'SAM (Severe Acute Malnutrition, n={class_stats["sam"]["count"]})'
}

def draw_publication_wireframe(ax, coords, lw=2.0, alpha=0.9, show_legend=False, color_override=None):
    u_idx = [lm_map[n] for n in upper_sequence]
    l_idx = [lm_map[n] for n in lower_sequence]
    
    u_color = color_override if color_override else '#e74c3c'
    l_color = color_override if color_override else '#2980b9'
    p_color = color_override if color_override else '#27ae60'
    v_color = color_override if color_override else '#7f8c8d'
    
    ax.plot(coords[u_idx, 0], coords[u_idx, 1], color=u_color, linestyle='-', linewidth=lw, alpha=alpha, label='Upper edge' if show_legend else '')
    ax.plot(coords[l_idx, 0], coords[l_idx, 1], color=l_color, linestyle='-', linewidth=lw, alpha=alpha, label='Lower edge' if show_legend else '')
    
    l1_i, l12_i, l4_i = lm_map[1], lm_map[12], lm_map[4]
    ax.plot([coords[l1_i, 0], coords[l12_i, 0]], [coords[l1_i, 1], coords[l12_i, 1]], color=p_color, linestyle='-', linewidth=lw, alpha=alpha, label='Landmark 1->12 Connection' if show_legend else '')
    ax.plot([coords[l1_i, 0], coords[l4_i, 0]], [coords[l1_i, 1], coords[l4_i, 1]], color=p_color, linestyle='--', linewidth=lw, alpha=alpha, label='Landmark 1->4 Connection' if show_legend else '')
    
    l20_i, l19_i, l3_i = lm_map[20], lm_map[19], lm_map[3]
    ax.plot([coords[l20_i, 0], coords[l3_i, 0]], [coords[l20_i, 1], coords[l3_i, 1]], color=p_color, linestyle='-', linewidth=lw*0.9, alpha=alpha)
    ax.plot([coords[l19_i, 0], coords[l3_i, 0]], [coords[l19_i, 1], coords[l3_i, 1]], color=p_color, linestyle='-', linewidth=lw*0.9, alpha=alpha)
    
    first_v = True
    for u_n, l_n in vertical_pairs:
        ui, li = lm_map[u_n], lm_map[l_n]
        ax.plot([coords[ui, 0], coords[li, 0]], [coords[ui, 1], coords[li, 1]], 
                color=v_color, linestyle='--', linewidth=lw*0.6, alpha=0.7*alpha, label='Pairwise Vertical Cross-sections' if (first_v and show_legend) else '')
        first_v = False

# --- FIGURE 1: 19-Landmark Cartesian Wireframe Skeleton ---
fig1, ax1 = plt.subplots(figsize=(13, 6))
mean_lm_phys = mean_shape * mean_cs_global

draw_publication_wireframe(ax1, mean_lm_phys, lw=2.2, alpha=0.95, show_legend=True)

# Main Landmarks
ax1.scatter(mean_lm_phys[lm_map[1], 0], mean_lm_phys[lm_map[1], 1], color='#e74c3c', s=160, zorder=6, edgecolors='black', linewidth=1.5, label='Shoulder (L1)')
ax1.scatter(mean_lm_phys[lm_map[2], 0], mean_lm_phys[lm_map[2], 1], color='#f39c12', s=160, zorder=6, edgecolors='black', linewidth=1.5, label='Elbow (L2)')
ax1.scatter(mean_lm_phys[lm_map[3], 0], mean_lm_phys[lm_map[3], 1], color='#2980b9', s=160, zorder=6, edgecolors='black', linewidth=1.5, label='Wrist (L3)')
ax1.scatter(mean_lm_phys[lm_map[4], 0], mean_lm_phys[lm_map[4], 1], color='#27ae60', s=160, zorder=6, edgecolors='black', linewidth=1.5, label='Armpit (L4)')

# Semi-landmarks
semi_indices = [lm_map[n] for n in lm_nums if n not in MAIN_LM_NUMS]
ax1.scatter(mean_lm_phys[semi_indices, 0], mean_lm_phys[semi_indices, 1], color='#8e44ad', s=55, zorder=5, edgecolors='black', linewidth=0.8, label='15 Semi-landmarks')

for num in lm_nums:
    idx = lm_map[num]
    x, y = mean_lm_phys[idx]
    if num in MAIN_LM_NUMS:
        ax1.annotate(f'{num} ({MAIN_LM_NAMES[num].split()[0]})', (x, y), textcoords="offset points", xytext=(0, 11),
                     ha='center', fontsize=9, fontweight='bold', color='#1a5276',
                     bbox=dict(boxstyle="round,pad=0.3", fc="#fcf3cf", ec="#f39c12", lw=1.2, alpha=0.95))
    else:
        ax1.annotate(f'{num}', (x, y), textcoords="offset points", xytext=(3, 5),
                     ha='left', fontsize=8, fontweight='bold', color='#c0392b' if num in upper_sequence else '#1f618d')

ax1.set_title("Cartesian Landmark Plot: Exact Pairwise Wireframe & Vertical Cross-sections", fontsize=14, fontweight='bold', pad=12)
ax1.set_xlabel("X (cm) -> right", fontweight='bold')
ax1.set_ylabel("Y (cm) -> up", fontweight='bold')
ax1.set_aspect('equal')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, shadow=True)
plt.tight_layout()
fig1_path = os.path.join(TARGET_DIR, 'fig1_wireframe_skeleton.png')
fig1.savefig(fig1_path, dpi=300)
plt.close(fig1)

# --- FIGURE 2: 4-Panel Class-Wise Point Cloud Scatter & Consensus Wireframe Overlay ---
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey=True)

ax_onc = axes2[0, 0]
onc_phys_all = class_stats['onc']['aligned_shapes'] * class_stats['onc']['centroid_sizes'][:, np.newaxis, np.newaxis]
for i in range(len(onc_phys_all)):
    ax_onc.scatter(onc_phys_all[i, :, 0], onc_phys_all[i, :, 1], color=colors['onc'], alpha=0.08, s=12, zorder=1)
m_onc = class_stats['onc']['mean_phys']
draw_publication_wireframe(ax_onc, m_onc, lw=2.2, alpha=0.95, color_override=colors['onc'])
ax_onc.scatter(m_onc[:, 0], m_onc[:, 1], color='black', s=45, zorder=4, edgecolors='white')
ax_onc.set_title(f"(a) ONC Landmark Point Cloud Scatter & Wireframe (n={class_stats['onc']['count']})", fontweight='bold', fontsize=12)
ax_onc.grid(True, linestyle='--', alpha=0.5)
ax_onc.set_aspect('equal')

ax_mam = axes2[0, 1]
mam_phys_all = class_stats['mam']['aligned_shapes'] * class_stats['mam']['centroid_sizes'][:, np.newaxis, np.newaxis]
for i in range(len(mam_phys_all)):
    ax_mam.scatter(mam_phys_all[i, :, 0], mam_phys_all[i, :, 1], color=colors['mam'], alpha=0.15, s=14, zorder=1)
m_mam = class_stats['mam']['mean_phys']
draw_publication_wireframe(ax_mam, m_mam, lw=2.2, alpha=0.95, color_override=colors['mam'])
ax_mam.scatter(m_mam[:, 0], m_mam[:, 1], color='black', s=45, zorder=4, edgecolors='white')
ax_mam.set_title(f"(b) MAM Clean Point Cloud Scatter & Wireframe (n={class_stats['mam']['count']})", fontweight='bold', fontsize=12)
ax_mam.grid(True, linestyle='--', alpha=0.5)
ax_mam.set_aspect('equal')

ax_sam = axes2[1, 0]
sam_phys_all = class_stats['sam']['aligned_shapes'] * class_stats['sam']['centroid_sizes'][:, np.newaxis, np.newaxis]
for i in range(len(sam_phys_all)):
    ax_sam.scatter(sam_phys_all[i, :, 0], sam_phys_all[i, :, 1], color=colors['sam'], alpha=0.15, s=14, zorder=1)
m_sam = class_stats['sam']['mean_phys']
draw_publication_wireframe(ax_sam, m_sam, lw=2.2, alpha=0.95, color_override=colors['sam'])
ax_sam.scatter(m_sam[:, 0], m_sam[:, 1], color='black', s=45, zorder=4, edgecolors='white')
ax_sam.set_title(f"(c) SAM Clean Point Cloud Scatter & Wireframe (n={class_stats['sam']['count']})", fontweight='bold', fontsize=12)
ax_sam.grid(True, linestyle='--', alpha=0.5)
ax_sam.set_aspect('equal')

ax_comp = axes2[1, 1]
for cls in ['onc', 'mam', 'sam']:
    m_cls = class_stats[cls]['mean_shape'] * mean_cs_global
    u_idx = [lm_map[n] for n in upper_sequence]
    l_idx = [lm_map[n] for n in lower_sequence]
    ax_comp.plot(m_cls[u_idx, 0], m_cls[u_idx, 1], color=colors[cls], linestyle='-', linewidth=2.5, alpha=0.9, label=labels[cls])
    ax_comp.plot(m_cls[l_idx, 0], m_cls[l_idx, 1], color=colors[cls], linestyle='-', linewidth=2.5, alpha=0.9)
    for u_n, l_n in vertical_pairs:
        ui, li = lm_map[u_n], lm_map[l_n]
        ax_comp.plot([m_cls[ui, 0], m_cls[li, 0]], [m_cls[ui, 1], m_cls[li, 1]], color=colors[cls], linestyle='--', linewidth=0.8, alpha=0.6)
    ax_comp.scatter(m_cls[:, 0], m_cls[:, 1], color=colors[cls], s=40, edgecolors='black', linewidth=0.8, zorder=4)

ax_comp.set_title("(d) Combined Consensus Wireframes Overlay", fontweight='bold', fontsize=12)
ax_comp.grid(True, linestyle='--', alpha=0.5)
ax_comp.set_aspect('equal')
ax_comp.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

for ax in axes2[1, :]:
    ax.set_xlabel("X (cm) -> right", fontweight='bold')
for ax in axes2[:, 0]:
    ax.set_ylabel("Y (cm) -> up", fontweight='bold')

plt.suptitle("Class-Wise Landmark Point Cloud Scatter & Consensus Wireframe Analysis", fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
fig2_path = os.path.join(TARGET_DIR, 'fig2_class_consensus_shapes.png')
fig2.savefig(fig2_path, dpi=300)
plt.close(fig2)

# --- FIGURE 3: Landmark Placement Error & Dispersion Ellipses ---
fig3, ax3 = plt.subplots(figsize=(13, 6.5))
mean_all_phys = mean_shape * mean_cs_global

draw_publication_wireframe(ax3, mean_all_phys, lw=1.2, alpha=0.4, show_legend=False)

for cls in ['onc', 'mam', 'sam']:
    res = class_stats[cls]['res_phys']
    m_shape = class_stats[cls]['mean_phys']
    for i in range(p):
        cov = np.cov(res[:, i, 0], res[:, i, 1])
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * 2 * np.sqrt(np.maximum(vals, 1e-6))
        
        ellipse = patches.Ellipse(xy=m_shape[i], width=width, height=height, angle=theta,
                                  edgecolor=colors[cls], facecolor=colors[cls], alpha=0.18, linewidth=1.2)
        ax3.add_patch(ellipse)

for cls in ['onc', 'mam', 'sam']:
    ax3.scatter(class_stats[cls]['mean_phys'][:, 0], class_stats[cls]['mean_phys'][:, 1],
                color=colors[cls], s=65, edgecolors='black', label=f'{labels[cls]} Mean (95% CI Ellipse)')

ax3.set_title("Landmark Placement & Shape Variability Ellipses (Clean ONC, MAM & SAM 95% Confidence Regions)", fontsize=14, fontweight='bold', pad=12)
ax3.set_xlabel("X Coordinate", fontweight='bold')
ax3.set_ylabel("Y Coordinate", fontweight='bold')
ax3.set_aspect('equal')
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, shadow=True)
plt.tight_layout()
fig3_path = os.path.join(TARGET_DIR, 'fig3_landmark_variability_ellipses.png')
fig3.savefig(fig3_path, dpi=300)
plt.close(fig3)

# --- FIGURE 4: Generic vs Data-Driven Synthetic Augmentation Comparison ---
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(15, 6.5))

np.random.seed(42)
mean_base = mean_shape * mean_cs_global
n_synth = 30

for _ in range(n_synth):
    beta_gen = np.random.uniform(0.1, 2.8)
    rot_gen = np.random.uniform(0, 2*np.pi)
    trans_gen = np.random.uniform(-5, 5, size=2)
    noise_gen = np.random.normal(0, 2.5, size=mean_base.shape)
    
    R = np.array([[np.cos(rot_gen), -np.sin(rot_gen)], [np.sin(rot_gen), np.cos(rot_gen)]])
    synth_gen = (mean_base @ R.T) * beta_gen + trans_gen + noise_gen
    
    draw_publication_wireframe(ax4a, synth_gen, lw=0.7, alpha=0.25, show_legend=False)

ax4a.set_title("Generic Simulation Cloud\n(Arbitrary U(0,10), U(0,2π), U(-5,5))", fontsize=13, fontweight='bold', color='#c0392b', pad=10)
ax4a.set_xlabel("X Coordinate", fontweight='bold')
ax4a.set_ylabel("Y Coordinate", fontweight='bold')
ax4a.grid(True, linestyle='--', alpha=0.5)

for _ in range(n_synth):
    beta_emp = np.random.uniform(clean_df['Scale_Ratio'].quantile(0.05), clean_df['Scale_Ratio'].quantile(0.95))
    rot_emp = np.radians(np.random.uniform(clean_df['Rel_Orientation_Deg'].quantile(0.05), clean_df['Rel_Orientation_Deg'].quantile(0.95)))
    trans_emp = np.array([np.random.uniform(clean_df['Trans_X'].quantile(0.05), clean_df['Trans_X'].quantile(0.95)),
                          np.random.uniform(clean_df['Trans_Y'].quantile(0.05), clean_df['Trans_Y'].quantile(0.95))])
    noise_emp = np.random.normal(0, intra_rmse, size=mean_base.shape)
    
    R = np.array([[np.cos(rot_emp), -np.sin(rot_emp)], [np.sin(rot_emp), np.cos(rot_emp)]])
    synth_emp = (mean_base @ R.T) * beta_emp + trans_emp + noise_emp
    
    draw_publication_wireframe(ax4b, synth_emp, lw=0.9, alpha=0.35, show_legend=False)

ax4b.set_title(f"Data-Driven Empirical Cloud (Clean Dataset)\n(Empirical Scale, Rotation, Translation & σ={intra_rmse:.2f})", fontsize=13, fontweight='bold', color='#27ae60', pad=10)
ax4b.set_xlabel("X Coordinate", fontweight='bold')
ax4b.set_ylabel("Y Coordinate", fontweight='bold')
ax4b.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
fig4_path = os.path.join(TARGET_DIR, 'fig4_synthetic_augmentation_comparison.png')
fig4.savefig(fig4_path, dpi=300)
plt.close(fig4)

# --- FIGURE 5: Empirical Parameter Distributions (Clean Dataset) ---
fig5, axes = plt.subplots(2, 2, figsize=(14, 10.5))

sns.boxplot(data=clean_df, x='Class_Type', y='Scale_Ratio', hue='Class_Type', palette=colors, legend=False, ax=axes[0, 0])
axes[0, 0].set_title("Empirical Scale Ratio (β) Distribution (Clean Dataset)", fontweight='bold', pad=10)
axes[0, 0].set_xlabel("Class Type", fontweight='bold')
axes[0, 0].set_ylabel("Scale Ratio (β)", fontweight='bold')
axes[0, 0].axhline(1.0, color='gray', linestyle='--', alpha=0.7)

sns.boxplot(data=clean_df, x='Class_Type', y='Rel_Orientation_Deg', hue='Class_Type', palette=colors, legend=False, ax=axes[0, 1])
axes[0, 1].set_title("Empirical Rotation (θ) Relative to Mean Pose (Clean Dataset)", fontweight='bold', pad=10)
axes[0, 1].set_xlabel("Class Type", fontweight='bold')
axes[0, 1].set_ylabel("Relative Angle (Degrees)", fontweight='bold')

sns.kdeplot(data=clean_df, x='Trans_X', hue='Class_Type', palette=colors, ax=axes[1, 0], fill=True, alpha=0.35)
axes[1, 0].set_title("Centroid Translation X (γ_x) Density (Clean Dataset)", fontweight='bold', pad=10)
axes[1, 0].set_xlabel("X Frame Shift (Units)", fontweight='bold')
axes[1, 0].set_ylabel("Density", fontweight='bold')

sns.histplot(all_intra_res.flatten(), kde=True, color='#8e44ad', ax=axes[1, 1], bins=40, stat="density")
axes[1, 1].set_title(f"Intra-Subject Digitization Noise (Clean Dataset, σ={intra_rmse:.3f})", fontweight='bold', pad=10)
axes[1, 1].set_xlabel("Landmark Digitization Error (Units)", fontweight='bold')
axes[1, 1].set_ylabel("Density", fontweight='bold')

plt.tight_layout()
fig5_path = os.path.join(TARGET_DIR, 'fig5_empirical_distributions.png')
fig5.savefig(fig5_path, dpi=300)
plt.close(fig5)

# --- 7. SYNTHETIC CLASS-SPECIFIC GENERATION & CSV EXPORT ---
print("\n=== GENERATING CLASS-SPECIFIC SYNTHETIC DATASETS & EXPORTING CSVS ===")

def generate_class_synthetic(class_type, num_samples=500, seed=42):
    np.random.seed(seed)
    p_cfg = class_stats[class_type]
    base_shape = p_cfg['mean_phys']
    p_count = len(base_shape)
    
    synth = np.zeros((num_samples, p_count, 2))
    for i in range(num_samples):
        beta = np.random.uniform(p_cfg['scale_ratio']['p5'], p_cfg['scale_ratio']['p95'])
        theta_rad = np.radians(np.random.uniform(p_cfg['rotation_deg']['p5'], p_cfg['rotation_deg']['p95']))
        R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)], [np.sin(theta_rad), np.cos(theta_rad)]])
        gamma = np.array([np.random.uniform(p_cfg['trans_x']['p5'], p_cfg['trans_x']['p95']),
                          np.random.uniform(p_cfg['trans_y']['p5'], p_cfg['trans_y']['p95'])])
        noise = np.random.normal(0, p_cfg['gpa_rmse'], size=(p_count, 2))
        synth[i] = (base_shape @ R.T) * beta + gamma + noise
    return synth

def export_synthetic_csv(synth_tensor, class_name, filename):
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
    out_path = os.path.join(TARGET_DIR, filename)
    out_df.to_csv(out_path, index=False)
    print(f"Exported '{filename}' ({len(out_df)} synthetic samples).")
    return out_df

synth_onc = generate_class_synthetic('onc', 500)
synth_mam = generate_class_synthetic('mam', 500)
synth_sam = generate_class_synthetic('sam', 500)

df_s_onc = export_synthetic_csv(synth_onc, 'onc', 'synthetic_augmented_onc_landmarks.csv')
df_s_mam = export_synthetic_csv(synth_mam, 'mam', 'synthetic_augmented_mam_landmarks.csv')
df_s_sam = export_synthetic_csv(synth_sam, 'sam', 'synthetic_augmented_sam_landmarks.csv')

comb_df = pd.concat([df_s_onc, df_s_mam, df_s_sam], ignore_index=True)
comb_path = os.path.join(TARGET_DIR, 'synthetic_augmented_balanced_dataset_combined.csv')
comb_df.to_csv(comb_path, index=False)
print(f"Exported 'synthetic_augmented_balanced_dataset_combined.csv' ({len(comb_df)} total balanced samples).")

# --- FIGURE 6: Generated Synthetic Data Wireframes (4-Panel Subplots) ---
fig6, axes6 = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey=True)

s_data = {'onc': synth_onc, 'mam': synth_mam, 'sam': synth_sam}

for idx, cls in enumerate(['onc', 'mam', 'sam']):
    r = idx // 2
    c = idx % 2
    ax = axes6[r, c]
    
    data = s_data[cls]
    for i in range(min(100, len(data))):
        ax.scatter(data[i, :, 0], data[i, :, 1], color=colors[cls], alpha=0.08, s=12, zorder=1)
        
    m_shape = data.mean(axis=0)
    draw_publication_wireframe(ax, m_shape, lw=2.2, alpha=0.95, color_override=colors[cls])
    ax.scatter(m_shape[:, 0], m_shape[:, 1], color='black', s=45, zorder=4, edgecolors='white')
    ax.set_title(f'({chr(97+idx)}) Generated Synthetic {cls.upper()} Wireframes (n=500)', fontweight='bold', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')

ax_d6 = axes6[1, 1]
for cls in ['onc', 'mam', 'sam']:
    m_shape = s_data[cls].mean(axis=0)
    u_idx = [lm_map[n] for n in upper_sequence]
    l_idx = [lm_map[n] for n in lower_sequence]
    ax_d6.plot(m_shape[u_idx, 0], m_shape[u_idx, 1], color=colors[cls], linestyle='-', linewidth=2.5, alpha=0.9, label=labels[cls])
    ax_d6.plot(m_shape[l_idx, 0], m_shape[l_idx, 1], color=colors[cls], linestyle='-', linewidth=2.5, alpha=0.9)
    for u_n, l_n in vertical_pairs:
        ui, li = lm_map[u_n], lm_map[l_n]
        ax_d6.plot([m_shape[ui, 0], m_shape[li, 0]], [m_shape[ui, 1], m_shape[li, 1]], color=colors[cls], linestyle='--', linewidth=0.8, alpha=0.6)
    ax_d6.scatter(m_shape[:, 0], m_shape[:, 1], color=colors[cls], s=40, edgecolors='black', linewidth=0.8, zorder=4)

ax_d6.set_title('(d) Combined Generated Synthetic Wireframes Overlay', fontweight='bold', fontsize=12)
ax_d6.grid(True, linestyle='--', alpha=0.5)
ax_d6.set_aspect('equal')
ax_d6.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

for ax in axes6[1, :]:
    ax.set_xlabel('X (cm) -> right', fontweight='bold')
for ax in axes6[:, 0]:
    ax.set_ylabel('Y (cm) -> up', fontweight='bold')

plt.suptitle('Generated Data-Driven Synthetic Landmark Wireframes & Point Cloud Analysis', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()

fig6_path = os.path.join(TARGET_DIR, 'fig6_generated_synthetic_wireframes.png')
fig6.savefig(fig6_path, dpi=300)
plt.close(fig6)

print("Saved fig6_generated_synthetic_wireframes.png successfully!")

# Save parameters to JSON summary file
summary_dict = {
    'dataset_cleaning': {
        'total_raw_samples': len(df),
        'inverted_distorted_samples_removed': int(df['Is_Inverted'].sum()),
        'clean_quality_samples_retained': int(len(clean_df)),
        'clean_class_counts': {
            'onc': int(class_stats['onc']['count']),
            'mam': int(class_stats['mam']['count']),
            'sam': int(class_stats['sam']['count'])
        }
    },
    'pairwise_wireframe': {
        'landmark_numbers': lm_nums,
        'main_landmarks': {'1': 'Shoulder', '2': 'Elbow', '3': 'Wrist', '4': 'Armpit'},
        'upper_edge_sequence': upper_sequence,
        'lower_edge_sequence': lower_sequence,
        'vertical_pairs': vertical_pairs
    },
    'comparison_table_dynamic': comp_data_dynamic,
    'overall_clean': {
        'count': int(N),
        'scale_ratio': {'min': float(clean_df['Scale_Ratio'].min()), 'max': float(clean_df['Scale_Ratio'].max()), 'mean': float(clean_df['Scale_Ratio'].mean()), 'std': float(clean_df['Scale_Ratio'].std()), 'p5': float(clean_df['Scale_Ratio'].quantile(0.05)), 'p95': float(clean_df['Scale_Ratio'].quantile(0.95))},
        'rotation_deg': {'min': float(clean_df['Orientation_Deg'].min()), 'max': float(clean_df['Orientation_Deg'].max()), 'mean': float(clean_df['Orientation_Deg'].mean()), 'std': float(clean_df['Orientation_Deg'].std()), 'p5': float(clean_df['Orientation_Deg'].quantile(0.05)), 'p95': float(clean_df['Orientation_Deg'].quantile(0.95))},
        'trans_x': {'min': float(clean_df['Trans_X'].min()), 'max': float(clean_df['Trans_X'].max()), 'mean': float(clean_df['Trans_X'].mean()), 'std': float(clean_df['Trans_X'].std()), 'p5': float(clean_df['Trans_X'].quantile(0.05)), 'p95': float(clean_df['Trans_X'].quantile(0.95))},
        'trans_y': {'min': float(clean_df['Trans_Y'].min()), 'max': float(clean_df['Trans_Y'].max()), 'mean': float(clean_df['Trans_Y'].mean()), 'std': float(clean_df['Trans_Y'].std()), 'p5': float(clean_df['Trans_Y'].quantile(0.05)), 'p95': float(clean_df['Trans_Y'].quantile(0.95))},
        'intra_subject_rmse': float(intra_rmse),
        'intra_subject_std_x': float(intra_std_x),
        'intra_subject_std_y': float(intra_std_y),
        'global_gpa_rmse': float(global_rmse)
    },
    'classes_clean': {
        cls: {
            'count': int(class_stats[cls]['count']),
            'scale_ratio': {k: float(v) for k, v in class_stats[cls]['scale_ratio'].items()},
            'rotation_deg': {k: float(v) for k, v in class_stats[cls]['rotation_deg'].items()},
            'trans_x': {k: float(v) for k, v in class_stats[cls]['trans_x'].items()},
            'trans_y': {k: float(v) for k, v in class_stats[cls]['trans_y'].items()},
            'gpa_rmse': float(class_stats[cls]['gpa_rmse']),
            'std_x': float(class_stats[cls]['std_x']),
            'std_y': float(class_stats[cls]['std_y'])
        } for cls in ['onc', 'mam', 'sam']
    },
    'synthetic_export_summary': {
        'synthetic_onc_samples': 500,
        'synthetic_mam_samples': 500,
        'synthetic_sam_samples': 500,
        'total_combined_samples': 1500,
        'files_generated': [
            'synthetic_augmented_onc_landmarks.csv',
            'synthetic_augmented_mam_landmarks.csv',
            'synthetic_augmented_sam_landmarks.csv',
            'synthetic_augmented_balanced_dataset_combined.csv',
            'fig6_generated_synthetic_wireframes.png'
        ]
    }
}

json_summary_path = os.path.join(TARGET_DIR, 'empirical_augmentation_parameters_summary.json')
with open(json_summary_path, 'w') as f:
    json.dump(summary_dict, f, indent=2)

print("Saved updated empirical_augmentation_parameters_summary.json with dynamic table values successfully!")
