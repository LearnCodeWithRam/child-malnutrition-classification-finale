import json

nb_path = r"GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/SAM_MAM_ONC_Out_of_Sample_Classification.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_cell_source = [
    "# STEP 3: Scaled Landmark Scatter Plot (Centroid Size Normalized)\n",
    "n_samples = M.shape[2] if 'M' in locals() or 'M' in globals() else n\n",
    "\n",
    "# Ensure centered landmark tensor exists\n",
    "if 'M_centered' not in locals() and 'M_centered' not in globals():\n",
    "    M_centered = np.zeros_like(M)\n",
    "    for i in range(n_samples):\n",
    "        M_centered[:, :, i], _ = center_shape(M[:, :, i])\n",
    "\n",
    "M_scaled = np.zeros_like(M_centered)\n",
    "for i in range(n_samples):\n",
    "    M_scaled[:, :, i], _ = scale_shape(M_centered[:, :, i])\n",
    "\n",
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
    "\n",
    "for ax, cls, title in zip(axes, classes, titles):\n",
    "    indices = np.where(true_class == cls)[0]\n",
    "    plot_arm_wireframe(ax, M_scaled[:, :, indices], color=class_colors[cls], \n",
    "                       alpha_solid=0.15, alpha_dashed=0.08, lw_solid=0.9, lw_dashed=0.6, alpha_scatter=0.25, s=12)\n",
    "    \n",
    "    # Overlay mean shape wireframe & centroid marker\n",
    "    cls_mean = np.mean(M_scaled[:, :, indices], axis=2)\n",
    "    plot_arm_wireframe(ax, cls_mean, color='black', \n",
    "                       alpha_solid=1.0, alpha_dashed=0.8, lw_solid=2.5, lw_dashed=1.5, alpha_scatter=1.0, s=60, label='Class Mean Wireframe')\n",
    "    ax.scatter(0, 0, color='black', s=80, marker='X', label='Centroid (0,0)', zorder=10)\n",
    "    \n",
    "    ax.set_title(f'Step 3: Scaled Arm Wireframes\\n{title} (n={len(indices)})', fontsize=12, fontweight='bold')\n",
    "    ax.set_xlabel('Scaled X coordinate')\n",
    "    ax.set_ylabel('Scaled Y coordinate')\n",
    "    ax.legend(loc='upper right')\n",
    "    ax.grid(True, alpha=0.3)\n",
    "    ax.axis('equal')\n",
    "\n",
    "plt.suptitle('Step 3: Centroid-Size Scaled Landmarks & Wireframes (Scale Removed)', fontsize=15, fontweight='bold', y=1.02)\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

cell_updated = False
for idx, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        src_text = "".join(cell.get("source", []))
        if "STEP 3: Scaled Landmark Scatter Plot" in src_text:
            cell["source"] = new_cell_source
            cell["outputs"] = []
            cell["execution_count"] = None
            cell_updated = True
            print(f"Successfully updated cell index {idx} in notebook!")
            break

if cell_updated:
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print("Notebook SAM_MAM_ONC_Out_of_Sample_Classification.ipynb updated and saved successfully!")
else:
    print("Error: Could not locate cell with STEP 3 in notebook.")
