import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

path = r'e:\ap-govt-EDA-model-tainning-testing\GM_out_of_sample_classification-main\GM_out_of_sample_classification-ap-govt\SAM_MAM_ONC_Out_of_Sample_Classification.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

globals_dict = {}

# Execute cells 1, 3, 5, 7, 8
for i in [1, 3, 5, 7, 8]:
    c = nb['cells'][i]
    src = ''.join(c['source'])
    exec(src, globals_dict)

# Now test updated Cell 9 code
test_code = """# STEP 3: Scaled Landmark Scatter Plot (Centroid Size Normalized)
n_samples = M.shape[2]

if 'M_centered' not in locals() and 'M_centered' not in globals():
    M_centered = np.zeros_like(M)
    for i in range(n_samples):
        M_centered[:, :, i], _ = center_shape(M[:, :, i])

M_scaled = np.zeros_like(M_centered)
for i in range(n_samples):
    M_scaled[:, :, i], _ = scale_shape(M_centered[:, :, i])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, cls, title in zip(axes, classes, titles):
    indices = np.where(true_class == cls)[0]
    plot_arm_wireframe(ax, M_scaled[:, :, indices], color=class_colors[cls], 
                       alpha_solid=0.15, alpha_dashed=0.08, lw_solid=0.9, lw_dashed=0.6, alpha_scatter=0.25, s=12)
    
    # Overlay mean shape wireframe & centroid marker
    cls_mean = np.mean(M_scaled[:, :, indices], axis=2)
    plot_arm_wireframe(ax, cls_mean, color='black', 
                       alpha_solid=1.0, alpha_dashed=0.8, lw_solid=2.5, lw_dashed=1.5, alpha_scatter=1.0, s=60, label='Class Mean Wireframe')
    ax.scatter(0, 0, color='black', s=80, marker='X', label='Centroid (0,0)', zorder=10)
    
    ax.set_title(f'Step 3: Scaled Arm Wireframes\\n{title} (n={len(indices)})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Scaled X coordinate')
    ax.set_ylabel('Scaled Y coordinate')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

plt.suptitle('Step 3: Centroid-Size Scaled Landmarks & Wireframes (Scale Removed)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
"""

exec(test_code, globals_dict)
plt.savefig(r'e:\ap-govt-EDA-model-tainning-testing\GM_out_of_sample_classification-main\GM_out_of_sample_classification-ap-govt\scratch\step3_plot_updated.png', bbox_inches='tight', dpi=150)
print("Updated Step 3 test code executed and saved step3_plot_updated.png successfully!")
