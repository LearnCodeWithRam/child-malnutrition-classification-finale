import json

path = r'GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/SAM_MAM_ONC_Out_of_Sample_Classification.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell23_source = [
    "# Filter dataset for SAM and ONC classes only\n",
    "df_2c = df[df['Class_Type'].isin(['sam', 'onc'])].copy().reset_index(drop=True)\n",
    "n_2c = len(df_2c)\n",
    "labels_2c = df_2c['Class_Type'].values\n",
    "y_binary = np.array([1 if l == 'sam' else 0 for l in labels_2c])\n",
    "\n",
    "# Extract raw feature matrix (38 landmark coordinates) for SAM vs ONC 2-class analysis\n",
    "X_raw = df_2c[x_cols + y_cols].values\n",
    "\n",
    "print('='*60, flush=True)\n",
    "print('2-CLASS DATASET OVERVIEW (SAM vs ONC)', flush=True)\n",
    "print('='*60, flush=True)\n",
    "print(f'Total Samples (N): {n_2c}', flush=True)\n",
    "print(f'Class Distribution: SAM = {sum(y_binary==1)}, ONC = {sum(y_binary==0)}', flush=True)\n",
    "print(f'Total Feature Coordinates: {len(x_cols) + len(y_cols)} (19 landmarks x 2 dimensions)', flush=True)\n"
]

nb["cells"][23]["source"] = cell23_source
nb["cells"][23]["outputs"] = []
nb["cells"][23]["execution_count"] = None

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Cell 23 updated with X_raw definition successfully!")
