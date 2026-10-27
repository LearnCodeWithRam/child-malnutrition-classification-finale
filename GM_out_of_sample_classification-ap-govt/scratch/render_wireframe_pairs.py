import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os

path = 'SAM_MAM_ONC_Out_of_Sample_Classification.ipynb'
if not os.path.exists(path):
    path = r'GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/SAM_MAM_ONC_Out_of_Sample_Classification.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

globals_dict = {}

# Execute cells 1, 3, 5, 7, 8, 10, 23, 35, 41
for idx in [1, 3, 5, 7, 8, 10, 23, 35, 41]:
    exec(''.join(nb['cells'][idx]['source']), globals_dict)

aligned_shapes_rot = globals_dict['aligned_shapes_rot']
true_class = globals_dict['true_class']
landmark_ids = globals_dict['landmark_ids']
id_to_idx = globals_dict['id_to_idx']
plot_arm_wireframe = globals_dict['plot_arm_wireframe']
df_grouped = globals_dict['df_grouped']

sam_indices = np.where(true_class == 'sam')[0]
onc_indices = np.where(true_class == 'onc')[0]

sam_mean = np.mean(aligned_shapes_rot[:, :, sam_indices], axis=2)
onc_mean = np.mean(aligned_shapes_rot[:, :, onc_indices], axis=2)

# Top selected landmark IDs from paired ranking
sorted_lids = df_grouped['Raw_ID'].values
top12_lids = set(sorted_lids[:12])

# Define Anatomical Transverse Cross-Section Landmark Pairs (Top-Down):
cross_section_pairs = [
    (1, 4),    # Shoulder
    (12, 13),  # Upper Arm
    (14, 13),  # Upper Arm / Mid-Arm
    (16, 15),  # Mid-Arm
    (18, 17),  # Elbow
    (10, 9),   # Upper Forearm
    (8, 7),    # Mid-Forearm
    (6, 5),    # Lower Forearm
    (20, 19)   # Wrist
]

fig, ax = plt.subplots(figsize=(14, 7))

# 1. Plot Complete Arm Wireframes for ONC (Healthy) and SAM (Malnutrition)
plot_arm_wireframe(ax, onc_mean, color='#2ecc71', alpha_solid=1.0, alpha_dashed=0.7, lw_solid=2.5, lw_dashed=1.5, alpha_scatter=0.9, s=50, label='Mean ONC Arm Wireframe (Healthy)')
plot_arm_wireframe(ax, sam_mean, color='#e74c3c', alpha_solid=1.0, alpha_dashed=0.7, lw_solid=2.5, lw_dashed=1.5, alpha_scatter=0.9, s=50, label='Mean SAM Arm Wireframe (Severe Malnutrition)')

# 2. Overlay Top-Down Transverse Cross-Section Pairs
for top_id, bot_id in cross_section_pairs:
    if top_id in id_to_idx and bot_id in id_to_idx:
        t_idx = id_to_idx[top_id]
        b_idx = id_to_idx[bot_id]
        
        is_top_selected = (top_id in top12_lids or bot_id in top12_lids)
        
        # Connect top-down transverse slice line on SAM mean shape
        tx, ty = sam_mean[t_idx, 0], sam_mean[t_idx, 1]
        bx, by = sam_mean[b_idx, 0], sam_mean[b_idx, 1]
        
        if is_top_selected:
            # Highlighted transverse cross-section slice
            ax.plot([tx, bx], [ty, by], color='#f1c40f', linestyle='-', linewidth=3.0, zorder=7)
            ax.scatter([tx, bx], [ty, by], color='#f1c40f', s=140, zorder=8, edgecolors='black', linewidths=1.5)
            # Annotate pair
            ax.annotate(f'L{top_id}', (tx, ty + 0.015), fontsize=10, fontweight='bold', color='#b7950b', ha='center')
            ax.annotate(f'L{bot_id}', (bx, by - 0.020), fontsize=10, fontweight='bold', color='#b7950b', ha='center')
        else:
            # Standard transverse slice line
            ax.plot([tx, bx], [ty, by], color='gray', linestyle=':', linewidth=1.2, zorder=4)

# Add custom legend entry for top selected pairs
ax.scatter([], [], color='#f1c40f', s=120, edgecolors='black', label='Top Selected 2D Cross-Section Pair (Top-Down)')

ax.set_title('Anatomical Arm Wireframe: Top Selected Transverse Cross-Section Pairs [(X_top, Y_top), (X_bot, Y_bot)]\nComparing Healthy ONC (Green) vs Severe Malnutrition SAM (Red)', fontsize=13, fontweight='bold')
ax.set_xlabel('GPA Aligned X Coordinate', fontsize=11)
ax.set_ylabel('GPA Aligned Y Coordinate', fontsize=11)
ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.axis('equal')

plt.tight_layout()
plt.savefig('scratch/wireframe_overlay_pairs.png', bbox_inches='tight', dpi=150)
print("Saved wireframe_overlay_pairs.png successfully!")
