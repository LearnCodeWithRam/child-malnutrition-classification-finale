import json

path = r'GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/SAM_MAM_ONC_Out_of_Sample_Classification.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell44_source = [
    "# Anatomical Arm Wireframe Overlay of Selected Top Transverse Cross-Section Landmark Pairs [(X_top, Y_top), (X_bot, Y_bot)]\n",
    "sorted_lids = df_grouped['Raw_ID'].values\n",
    "top12_lids = set(sorted_lids[:12])\n",
    "\n",
    "sam_indices = np.where(true_class == 'sam')[0]\n",
    "onc_indices = np.where(true_class == 'onc')[0]\n",
    "\n",
    "sam_mean = np.mean(aligned_shapes_rot[:, :, sam_indices], axis=2)\n",
    "onc_mean = np.mean(aligned_shapes_rot[:, :, onc_indices], axis=2)\n",
    "\n",
    "# Define Anatomical Transverse Cross-Section Landmark Pairs (Top-Down):\n",
    "cross_section_pairs = [\n",
    "    (1, 4),    # Shoulder Slice\n",
    "    (12, 13),  # Upper Arm Slice 1\n",
    "    (14, 13),  # Upper Arm Slice 2\n",
    "    (16, 15),  # Mid-Arm Slice\n",
    "    (18, 17),  # Elbow Slice\n",
    "    (10, 9),   # Upper Forearm Slice\n",
    "    (8, 7),    # Mid-Forearm Slice\n",
    "    (6, 5),    # Lower Forearm Slice\n",
    "    (20, 19)   # Wrist Slice\n",
    "]\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(14, 7))\n",
    "\n",
    "# 1. Plot Complete Arm Wireframes for ONC (Healthy) and SAM (Malnutrition)\n",
    "plot_arm_wireframe(ax, onc_mean, color='#2ecc71', alpha_solid=1.0, alpha_dashed=0.7, lw_solid=2.5, lw_dashed=1.5, alpha_scatter=0.9, s=50, label='Mean ONC Arm Wireframe (Healthy)')\n",
    "plot_arm_wireframe(ax, sam_mean, color='#e74c3c', alpha_solid=1.0, alpha_dashed=0.7, lw_solid=2.5, lw_dashed=1.5, alpha_scatter=0.9, s=50, label='Mean SAM Arm Wireframe (Severe Malnutrition)')\n",
    "\n",
    "# 2. Overlay Top-Down Transverse Cross-Section Pairs\n",
    "for top_id, bot_id in cross_section_pairs:\n",
    "    if top_id in id_to_idx and bot_id in id_to_idx:\n",
    "        t_idx = id_to_idx[top_id]\n",
    "        b_idx = id_to_idx[bot_id]\n",
    "        \n",
    "        is_top_selected = (top_id in top12_lids or bot_id in top12_lids)\n",
    "        \n",
    "        # Connect top-down transverse slice line on SAM mean shape\n",
    "        tx, ty = sam_mean[t_idx, 0], sam_mean[t_idx, 1]\n",
    "        bx, by = sam_mean[b_idx, 0], sam_mean[b_idx, 1]\n",
    "        \n",
    "        if is_top_selected:\n",
    "            # Highlighted transverse cross-section slice\n",
    "            ax.plot([tx, bx], [ty, by], color='#f1c40f', linestyle='-', linewidth=3.0, zorder=7)\n",
    "            ax.scatter([tx, bx], [ty, by], color='#f1c40f', s=140, zorder=8, edgecolors='black', linewidths=1.5)\n",
    "            # Annotate pair top-down\n",
    "            ax.annotate(f'L{top_id}', (tx, ty + 0.012), fontsize=10, fontweight='bold', color='#b7950b', ha='center')\n",
    "            ax.annotate(f'L{bot_id}', (bx, by - 0.016), fontsize=10, fontweight='bold', color='#b7950b', ha='center')\n",
    "        else:\n",
    "            # Standard transverse slice line\n",
    "            ax.plot([tx, bx], [ty, by], color='gray', linestyle=':', linewidth=1.2, zorder=4)\n",
    "\n",
    "# Custom legend entry for top selected pairs\n",
    "ax.scatter([], [], color='#f1c40f', s=120, edgecolors='black', label='Top Selected 2D Cross-Section Pair (Top-Down)')\n",
    "\n",
    "ax.set_title('Anatomical Arm Wireframe: Top Selected Transverse Cross-Section Pairs [(X_top, Y_top), (X_bot, Y_bot)]\\nComparing Healthy ONC (Green) vs Severe Malnutrition SAM (Red)', fontsize=13, fontweight='bold')\n",
    "ax.set_xlabel('GPA Aligned X Coordinate', fontsize=11)\n",
    "ax.set_ylabel('GPA Aligned Y Coordinate', fontsize=11)\n",
    "ax.legend(loc='upper right', fontsize=10, framealpha=0.95)\n",
    "ax.grid(True, alpha=0.3)\n",
    "ax.axis('equal')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

nb["cells"][44]["source"] = cell44_source
nb["cells"][44]["outputs"] = []
nb["cells"][44]["execution_count"] = None

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Cell 44 updated with anatomical wireframe top-down paired overlay successfully!")
