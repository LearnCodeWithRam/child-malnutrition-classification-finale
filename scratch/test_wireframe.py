import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv(r'e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/SAM-Photo/final_cleaned_dataset_full_with_muac.csv')
r = df.iloc[0]

fig, ax = plt.subplots(figsize=(10, 4))

upper_ids = [12, 14, 16, 18, 10, 8, 6, 20]
lower_ids = [4, 13, 15, 17, 9, 7, 5, 19]

# Upper edge (red)
ax.plot([r[f'X{i}'] for i in upper_ids], [r[f'Y{i}'] for i in upper_ids], 'r-', lw=2, label='Upper edge')
# Lower edge (blue)
ax.plot([r[f'X{i}'] for i in lower_ids], [r[f'Y{i}'] for i in lower_ids], 'b-', lw=2, label='Lower edge')
# Landmark 1 -> 12 (green)
ax.plot([r['X1'], r['X12']], [r['Y1'], r['Y12']], 'g-', lw=2.5, label='Landmark 1->12 Connection')
# Landmark 1 -> 4 (green)
ax.plot([r['X1'], r['X4']], [r['Y1'], r['Y4']], 'g-', lw=2.5, label='Landmark 1->11/4 Connection')
# Wrist boundary
ax.plot([r['X20'], r['X19']], [r['Y20'], r['Y19']], color='green', lw=1.5, label='Wrist Boundary')

# Cross sections (grey dashed)
cross_pairs = [[12, 4], [14, 13], [16, 15], [18, 17], [10, 9], [8, 7], [6, 5], [20, 19]]
for idx, (u, l) in enumerate(cross_pairs):
    lbl = 'Transverse Cross-Sections' if idx == 0 else None
    ax.plot([r[f'X{u}'], r[f'X{l}']], [r[f'Y{u}'], r[f'Y{l}']], 'k--', alpha=0.5, lw=1.2, label=lbl)

# Midline (1 -> 2 -> 3)
ax.plot([r['X1'], r['X2'], r['X3']], [r['Y1'], r['Y2'], r['Y3']], 'y-', lw=2.5, label='Center Midline (1->Elbow->Wrist)')

# Scatter points
ax.scatter([r['X1']], [r['Y1']], color='red', s=80, zorder=6, label='Landmark 1 (Shoulder)')
ax.scatter([r['X12'], r['X4']], [r['Y12'], r['Y4']], color='green', s=60, zorder=6)
ax.scatter([r['X2']], [r['Y2']], color='orange', s=70, zorder=6, label='Landmark 2 (Elbow)')
ax.scatter([r['X3']], [r['Y3']], color='blue', edgecolors='red', s=70, zorder=6, label='Landmark 3 (Wrist)')

for lid in [1, 12, 14, 16, 18, 10, 8, 6, 20, 4, 13, 15, 17, 9, 7, 5, 19, 2, 3]:
    lbl_text = '11/4' if lid == 4 else ('Elbow' if lid == 2 else ('Wrist' if lid == 3 else str(lid)))
    ax.text(r[f'X{lid}'] + 0.1, r[f'Y{lid}'] + 0.15, lbl_text, fontsize=8, color='#333333', fontweight='bold')

muac_val = r['muac_cm']
ax.set_title(f'Cartesian Landmark Plot | Shoulder->Wrist | MUAC: {muac_val:.2f} cm')
ax.set_xlabel('X (cm) -> right')
ax.set_ylabel('Y (cm) -> up')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig(r'e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/scratch/wireframe_test_plot.png', dpi=120)
print('Wireframe plot generated successfully!')
