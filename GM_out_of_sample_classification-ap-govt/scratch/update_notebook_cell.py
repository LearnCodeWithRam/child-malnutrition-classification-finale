import json
import os

nb_path = r"GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/SAM_MAM_ONC_Out_of_Sample_Classification.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_cell_lines = [
    "# Load dataset\n",
    'df_path = "heilight_v3 - Sheet1.csv"\n',
    "if not os.path.exists(df_path):\n",
    '    df_path = r"GM_out_of_sample_classification-main\\GM_out_of_sample_classification-ap-govt\\heilight_v3 - Sheet1.csv"\n',
    "\n",
    "df = pd.read_csv(df_path)\n",
    "\n",
    'print("="*60, flush=True)\n',
    'print("DATASET OVERVIEW", flush=True)\n',
    'print("="*60, flush=True)\n',
    'print(f"Loaded File: {df_path}", flush=True)\n',
    'print(f"Dataset Shape: {df.shape}", flush=True)\n',
    "classes_str = df['Class_Type'].unique()\n",
    'print(f"Classes Present: {classes_str}", flush=True)\n',
    'print("\\nClass Distribution:", flush=True)\n',
    "print(df['Class_Type'].value_counts(), flush=True)\n",
    "\n",
    "# Extract X and Y landmark columns\n",
    "x_cols = [c for c in df.columns if c.startswith('X') and c[1:].isdigit()]\n",
    "y_cols = [c for c in df.columns if c.startswith('Y') and c[1:].isdigit()]\n",
    "\n",
    "landmark_ids = [int(c[1:]) for c in x_cols]\n",
    "k = len(landmark_ids)\n",
    "n = len(df)\n",
    "\n",
    "# Map Landmark ID to 0-based array index\n",
    "id_to_idx = {lid: idx for idx, lid in enumerate(landmark_ids)}\n",
    "\n",
    "# Define Arm Wireframe Contour Connections:\n",
    "# 1. Upper Contour (Solid): 1 -> 12 -> 14 -> 16 -> 18 -> 10 -> 8 -> 6 -> 20\n",
    "upper_contour_ids = [1, 12, 14, 16, 18, 10, 8, 6, 20]\n",
    "upper_contour_idx = [id_to_idx[lid] for lid in upper_contour_ids]\n",
    "\n",
    "# 2. Lower Contour (Solid): 4 -> 13 -> 15 -> 17 -> 9 -> 7 -> 5 -> 19\n",
    "lower_contour_ids = [4, 13, 15, 17, 9, 7, 5, 19]\n",
    "lower_contour_idx = [id_to_idx[lid] for lid in lower_contour_ids]\n",
    "\n",
    "# 3. Outer Boundaries (Solid): Shoulder [1, 4], Wrist [20, 3, 19]\n",
    "shoulder_edge_idx = [id_to_idx[1], id_to_idx[4]]\n",
    "wrist_edge_idx = [id_to_idx[20], id_to_idx[3], id_to_idx[19]]\n",
    "\n",
    "# 4. Transverse Cross-Sections (Dashed): 12-4, 14-13, 16-15, 18-17, 10-9, 8-7, 6-5\n",
    "dashed_cross_ids = [\n",
    "    [12, 4],\n",
    "    [14, 13],\n",
    "    [16, 15],\n",
    "    [18, 17],\n",
    "    [10, 9],\n",
    "    [8, 7],\n",
    "    [6, 5]\n",
    "]\n",
    "dashed_cross_idx = [[id_to_idx[l1], id_to_idx[l2]] for l1, l2 in dashed_cross_ids]\n",
    "\n",
    "# 5. Midline Center Line (Dashed): 2 -> 3\n",
    "midline_idx = [id_to_idx[2], id_to_idx[3]]\n",
    "\n",
    'print(f"\\nTotal Landmark Points (k): {k}", flush=True)\n',
    'print(f"Landmark IDs: {landmark_ids}", flush=True)\n',
    'print("Solid Boundary Contours defined: Upper (9 pts), Lower (8 pts), Wrist/Shoulder edges", flush=True)\n',
    'print(f"Dashed Internal Cross-Sections defined: {len(dashed_cross_ids)} vertical links + Midline (2-3)", flush=True)\n',
    "\n",
    "# Convert DataFrame into 3D NumPy shape array (k, 2, n)\n",
    "M = np.zeros((k, 2, n))\n",
    "for i in range(n):\n",
    "    M[:, 0, i] = df.iloc[i][x_cols].values.astype(float)\n",
    "    M[:, 1, i] = df.iloc[i][y_cols].values.astype(float)\n",
    "\n",
    "true_class = df['Class_Type'].values\n",
    "class_colors = {'sam': '#e74c3c', 'mam': '#f39c12', 'onc': '#2ecc71'}\n",
    "\n",
    'print(f"\\n3D Landmark Tensor constructed: {M.shape} (k={k}, coords=2, samples={n})", flush=True)\n'
]

updated = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src_text = "".join(cell["source"])
        if "# Load dataset" in src_text or "df_path" in src_text:
            cell["source"] = new_cell_lines
            cell["outputs"] = []
            cell["execution_count"] = None
            updated = True
            print("Successfully updated notebook cell with auto-path resolution!")
            break

if updated:
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Notebook updated successfully.")
