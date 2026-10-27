import os
import json
import pandas as pd

# Paths
base_v3 = r"e:\ap-govt-EDA-model-tainning-testing\GM_out_of_sample_classification-main\GM_out_of_sample_classification-ap-govt\Experiment and Modeling\version3"
base_aug = r"e:\ap-govt-EDA-model-tainning-testing\GM_out_of_sample_classification-main\GM_out_of_sample_classification-ap-govt\Data-Augmentation-by-Geometric Transformation Modeling\train-test-split-and -augmentation"

# 1. Generate test_dataset_balanced_43.csv in both locations
test_orig_path = os.path.join(base_aug, "test_dataset_original.csv")
df_test_orig = pd.read_csv(test_orig_path)

df_sam_43 = df_test_orig[df_test_orig['Class_Type'] == 'sam']
df_mam_43 = df_test_orig[df_test_orig['Class_Type'] == 'mam'].sample(n=43, random_state=42)
df_onc_43 = df_test_orig[df_test_orig['Class_Type'] == 'onc'].sample(n=43, random_state=42)

df_test_bal43 = pd.concat([df_sam_43, df_mam_43, df_onc_43]).sample(frac=1, random_state=42).reset_index(drop=True)

csv_path_aug = os.path.join(base_aug, "test_dataset_balanced_43.csv")
csv_path_v3  = os.path.join(base_v3, "test_dataset_balanced_43.csv")

df_test_bal43.to_csv(csv_path_aug, index=False)
df_test_bal43.to_csv(csv_path_v3, index=False)

print(f"[Export] Saved test_dataset_balanced_43.csv ({df_test_bal43.shape[0]} rows) to both locations.")
print(df_test_bal43['Class_Type'].value_counts())

# 2. Update SAM_MAM_ONC_Balanced_Pipeline_Train_test.ipynb in version3
nb_path_v3 = os.path.join(base_v3, "SAM_MAM_ONC_Balanced_Pipeline_Train_test.ipynb")
with open(nb_path_v3, 'r', encoding='utf-8') as f:
    nb_v3 = json.load(f)

md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Section 10.1: Balanced 43-Sample Test Set Evaluation (43 ONC, 43 MAM, 43 SAM)\n",
        "\n",
        "To evaluate model performance on a perfectly balanced 1:1:1 out-of-sample test set matching the minority SAM sample count ($n=43$), we evaluate the trained classifiers on `df_test_bal_43` (43 ONC, 43 MAM, 43 SAM; total $n=129$).\n"
    ]
}

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Create 1:1:1 Balanced 43-Sample Test Subset (43 SAM, 43 MAM, 43 ONC -> Total n=129)\n",
        "df_te_sam_43 = df_test_raw[df_test_raw[target_col] == 'sam']\n",
        "df_te_mam_43 = df_test_raw[df_test_raw[target_col] == 'mam'].sample(n=43, random_state=RANDOM_STATE)\n",
        "df_te_onc_43 = df_test_raw[df_test_raw[target_col] == 'onc'].sample(n=43, random_state=RANDOM_STATE)\n",
        "\n",
        "df_test_bal_43 = pd.concat([df_te_sam_43, df_te_mam_43, df_te_onc_43]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)\n",
        "\n",
        "# Compute Allometric Residuals for 43-sample balanced test set\n",
        "df_tr_res_43, df_te_res_43, res_cols_43 = compute_allometric_residuals_df(df_train_bal, df_test_bal_43, gpa_cols)\n",
        "for col in res_cols_43:\n",
        "    df_test_bal_43[col] = df_te_res_43[col]\n",
        "\n",
        "X_test_43 = df_test_bal_43[feature_cols].values\n",
        "y_test_43 = df_test_bal_43[target_col].values\n",
        "y_test_43_num = np.array([label_map[y] for y in y_test_43])\n",
        "\n",
        "# Transform features using trained scaler\n",
        "X_te_43_s = scaler_final.transform(X_test_43)\n",
        "\n",
        "# Evaluate XGBoost on Balanced 43 Test Set\n",
        "y_pred_xgb_43 = xgb_final.predict(X_te_43_s)\n",
        "acc_xgb_43 = accuracy_score(y_test_43_num, y_pred_xgb_43) * 100\n",
        "macro_f1_xgb_43 = f1_score(y_test_43_num, y_pred_xgb_43, average='macro') * 100\n",
        "\n",
        "print('='*70)\n",
        "print('BALANCED 43-SAMPLE TEST EVALUATION (43 SAM, 43 MAM, 43 ONC)')\n",
        "print('='*70)\n",
        "print(f'Balanced Test Accuracy (43-sample)       : {acc_xgb_43:.2f}%')\n",
        "print(f'Balanced Test Macro F1-Score (43-sample) : {macro_f1_xgb_43:.2f}%')\n",
        "print('\\nClassification Report (43-sample Balanced Test):')\n",
        "print(classification_report(y_test_43_num, y_pred_xgb_43, target_names=['SAM', 'MAM', 'ONC']))\n",
        "\n",
        "# Evaluate Extra Trees on Balanced 43 Test Set\n",
        "y_pred_et_43 = et_final.predict(X_te_43_s)\n",
        "acc_et_43 = accuracy_score(y_test_43_num, y_pred_et_43) * 100\n",
        "\n",
        "# Plot Confusion Matrices for 43-sample Balanced Test Set\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 6))\n",
        "\n",
        "cm_xgb_43 = confusion_matrix(y_test_43_num, y_pred_xgb_43, labels=[0, 1, 2])\n",
        "sns.heatmap(cm_xgb_43, annot=True, fmt='d', cmap='Blues', ax=axes[0],\n",
        "            xticklabels=['SAM', 'MAM', 'ONC'], yticklabels=['SAM', 'MAM', 'ONC'],\n",
        "            annot_kws={'size': 14, 'weight': 'bold'})\n",
        "axes[0].set_title(f'XGBoost (43-Sample Balanced Test)\\nAccuracy: {acc_xgb_43:.2f}%', fontsize=13, fontweight='bold')\n",
        "axes[0].set_xlabel('Predicted Class', fontsize=11)\n",
        "axes[0].set_ylabel('True Class', fontsize=11)\n",
        "\n",
        "cm_et_43 = confusion_matrix(y_test_43_num, y_pred_et_43, labels=[0, 1, 2])\n",
        "sns.heatmap(cm_et_43, annot=True, fmt='d', cmap='Greens', ax=axes[1],\n",
        "            xticklabels=['SAM', 'MAM', 'ONC'], yticklabels=['SAM', 'MAM', 'ONC'],\n",
        "            annot_kws={'size': 14, 'weight': 'bold'})\n",
        "axes[1].set_title(f'Extra Trees (43-Sample Balanced Test)\\nAccuracy: {acc_et_43:.2f}%', fontsize=13, fontweight='bold')\n",
        "axes[1].set_xlabel('Predicted Class', fontsize=11)\n",
        "axes[1].set_ylabel('True Class', fontsize=11)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]
}

# Check if Section 10.1 cell already added
already_added = False
for cell in nb_v3['cells']:
    if cell['cell_type'] == 'markdown' and 'Section 10.1' in ''.join(cell['source']):
        already_added = True
        break

if not already_added:
    # Insert right before conclusion cell (which is the last cell)
    nb_v3['cells'].insert(-1, md_cell)
    nb_v3['cells'].insert(-1, code_cell)

    with open(nb_path_v3, 'w', encoding='utf-8') as f:
        json.dump(nb_v3, f, indent=1)
    print(f"[Updated] Notebook updated successfully: {nb_path_v3}")
else:
    print(f"[Info] Section 10.1 cell already present in: {nb_path_v3}")

# Also update Geometric_Transformation_Data_Augmentation_train_test.ipynb in train-test-split-and -augmentation
nb_path_aug = os.path.join(base_aug, "Geometric_Transformation_Data_Augmentation_train_test.ipynb")
with open(nb_path_aug, 'r', encoding='utf-8') as f:
    nb_aug = json.load(f)

md_cell_aug = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6. Balanced 43-Sample Test Dataset Export (1:1:1 Ratio Across Test Classes)\n",
        "\n",
        "To evaluate model performance on a perfectly balanced out-of-sample test set matching the minority SAM sample count ($n=43$), we sample 43 specimens from ONC and 43 specimens from MAM alongside the 43 SAM test specimens.\n"
    ]
}

code_cell_aug = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Create Balanced 43-Sample Test Set (43 ONC, 43 MAM, 43 SAM -> Total n=129)\n",
        "df_sam_test_43 = df_test_orig[df_test_orig['Class_Type'] == 'sam']\n",
        "df_mam_test_43 = df_test_orig[df_test_orig['Class_Type'] == 'mam'].sample(n=43, random_state=42).reset_index(drop=True)\n",
        "df_onc_test_43 = df_test_orig[df_test_orig['Class_Type'] == 'onc'].sample(n=43, random_state=42).reset_index(drop=True)\n",
        "\n",
        "df_test_balanced_43 = pd.concat([df_sam_test_43, df_mam_test_43, df_onc_test_43]).sample(frac=1, random_state=42).reset_index(drop=True)\n",
        "\n",
        "test_bal43_path = os.path.join(out_dir_path, \"test_dataset_balanced_43.csv\")\n",
        "df_test_balanced_43.to_csv(test_bal43_path, index=False)\n",
        "\n",
        "print('='*70)\n",
        "print('            BALANCED 43-SAMPLE TEST DATASET (1:1:1 RATIO)                ')\n",
        "print('='*70)\n",
        "print(f'Balanced Test Dataset Saved to: {test_bal43_path}')\n",
        "print(f'Total Specimen Count: {len(df_test_balanced_43)}')\n",
        "print(df_test_balanced_43['Class_Type'].value_counts().to_string())\n"
    ]
}

already_added_aug = False
for cell in nb_aug['cells']:
    if cell['cell_type'] == 'markdown' and 'Balanced 43-Sample Test' in ''.join(cell['source']):
        already_added_aug = True
        break

if not already_added_aug:
    nb_aug['cells'].append(md_cell_aug)
    nb_aug['cells'].append(code_cell_aug)
    with open(nb_path_aug, 'w', encoding='utf-8') as f:
        json.dump(nb_aug, f, indent=1)
    print(f"[Updated] Notebook updated successfully: {nb_path_aug}")
else:
    print(f"[Info] Section 6 cell already present in: {nb_path_aug}")
