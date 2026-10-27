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
    "# Extract raw feature matrix and scaled feature matrix (38 landmark coordinates) for SAM vs ONC 2-class analysis\n",
    "X_raw = df_2c[x_cols + y_cols].values\n",
    "X = X_raw  # Alias for backward compatibility\n",
    "scaler_2c = StandardScaler()\n",
    "X_scaled = scaler_2c.fit_transform(X_raw)\n",
    "feature_names = [f'X{i}' for i in landmark_ids] + [f'Y{i}' for i in landmark_ids]\n",
    "\n",
    "# Landmark coordinate matrices and class means for SAM vs ONC analysis\n",
    "X_x = df_2c[x_cols].values\n",
    "X_y = df_2c[y_cols].values\n",
    "sam_mask = (y_binary == 1)\n",
    "onc_mask = (y_binary == 0)\n",
    "mean_sam_x = np.mean(X_x[sam_mask], axis=0)\n",
    "mean_sam_y = np.mean(X_y[sam_mask], axis=0)\n",
    "mean_onc_x = np.mean(X_x[onc_mask], axis=0)\n",
    "mean_onc_y = np.mean(X_y[onc_mask], axis=0)\n",
    "\n",
    "# Multi-model dictionary for Feature Selection Comparison\n",
    "models_fs = {\n",
    "    'Linear Discriminant Analysis (LDA)': LinearDiscriminantAnalysis(),\n",
    "    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),\n",
    "    'Support Vector Machine (SVM RBF)': SVC(kernel='rbf', random_state=42),\n",
    "    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),\n",
    "    'Gradient Boosting': GradientBoostingClassifier(random_state=42)\n",
    "}\n",
    "\n",
    "print('='*60, flush=True)\n",
    "print('2-CLASS DATASET OVERVIEW (SAM vs ONC)', flush=True)\n",
    "print('='*60, flush=True)\n",
    "print(f'Total Samples (N): {n_2c}', flush=True)\n",
    "print(f'Class Distribution: SAM = {sum(y_binary==1)}, ONC = {sum(y_binary==0)}', flush=True)\n",
    "print(f'Total Feature Coordinates: {len(x_cols) + len(y_cols)} (19 landmarks x 2 dimensions)', flush=True)\n"
]

for idx, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "2-CLASS DATASET OVERVIEW" in src:
            cell["source"] = cell23_source
            cell["outputs"] = []
            cell["execution_count"] = None
            print(f"Cell {idx} updated successfully!")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Saved updated notebook.")
