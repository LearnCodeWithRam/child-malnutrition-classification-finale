import json

path = r'GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/SAM_MAM_ONC_Out_of_Sample_Classification.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell35_source = [
    "# Compute Feature Importance Metrics across 3 Methods: Lasso L1, Random Forest, ANOVA F-test\n",
    "if 'X_scaled' not in locals() and 'X_scaled' not in globals():\n",
    "    if 'X_raw' not in locals() and 'X_raw' not in globals():\n",
    "        df_2c = df[df['Class_Type'].isin(['sam', 'onc'])].copy().reset_index(drop=True)\n",
    "        y_binary = np.array([1 if l == 'sam' else 0 for l in df_2c['Class_Type'].values])\n",
    "        X_raw = df_2c[x_cols + y_cols].values\n",
    "    scaler_2c = StandardScaler()\n",
    "    X_scaled = scaler_2c.fit_transform(X_raw)\n",
    "\n",
    "if 'feature_names' not in locals() and 'feature_names' not in globals():\n",
    "    feature_names = [f'X{i}' for i in landmark_ids] + [f'Y{i}' for i in landmark_ids]\n",
    "\n",
    "rf = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "rf.fit(X_scaled, y_binary)\n",
    "rf_importances = rf.feature_importances_\n",
    "\n",
    "lr_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, random_state=42)\n",
    "lr_l1.fit(X_scaled, y_binary)\n",
    "lr_coefs = np.abs(lr_l1.coef_[0])\n",
    "\n",
    "f_vals, p_vals = f_classif(X_scaled, y_binary)\n",
    "\n",
    "df_imp = pd.DataFrame({\n",
    "    'Feature': feature_names,\n",
    "    'Lasso_L1_Weight': lr_coefs,\n",
    "    'RF_Importance': rf_importances,\n",
    "    'ANOVA_F_Score': f_vals,\n",
    "    'p_value': p_vals\n",
    "}).sort_values(by='Lasso_L1_Weight', ascending=False).reset_index(drop=True)\n",
    "\n",
    "print('TOP 20 RANKED MORPHOMETRIC LANDMARK FEATURES (SAM vs ONC):', flush=True)\n",
    "print('='*70, flush=True)\n",
    "print(df_imp.head(20).to_string(index=False), flush=True)\n"
]

for idx, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "Compute Feature Importance Metrics across 3 Methods" in src:
            cell["source"] = cell35_source
            cell["outputs"] = []
            cell["execution_count"] = None
            print(f"Successfully updated Cell index {idx} in notebook!")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Saved updated notebook.")
