import json

path = r'GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/SAM_MAM_ONC_Out_of_Sample_Classification.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell37_source = [
    "# Evaluate Feature Selection Performance Curve (k=2 to k=38)\n",
    "if 'X_raw' not in locals() and 'X_raw' not in globals():\n",
    "    df_2c = df[df['Class_Type'].isin(['sam', 'onc'])].copy().reset_index(drop=True)\n",
    "    y_binary = np.array([1 if l == 'sam' else 0 for l in df_2c['Class_Type'].values])\n",
    "    X_raw = df_2c[x_cols + y_cols].values\n",
    "\n",
    "X = X_raw  # Ensure X refers to X_raw\n",
    "\n",
    "k_features = [2, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35, 38]\n",
    "skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)\n",
    "sorted_indices = np.argsort(df_imp['Lasso_L1_Weight'].values)[::-1]\n",
    "\n",
    "lr_accs, rf_accs = [], []\n",
    "for k in k_features:\n",
    "    top_k_idx = sorted_indices[:k]\n",
    "    X_k = X_raw[:, top_k_idx]\n",
    "    acc_lr_list, acc_rf_list = [], []\n",
    "    for train_idx, test_idx in skf.split(X_k, y_binary):\n",
    "        sc = StandardScaler()\n",
    "        X_tr = sc.fit_transform(X_k[train_idx])\n",
    "        X_te = sc.transform(X_k[test_idx])\n",
    "        clf_lr = LogisticRegression(max_iter=1000, random_state=42)\n",
    "        clf_lr.fit(X_tr, y_binary[train_idx])\n",
    "        acc_lr_list.append(accuracy_score(y_binary[test_idx], clf_lr.predict(X_te)))\n",
    "        clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "        clf_rf.fit(X_tr, y_binary[train_idx])\n",
    "        acc_rf_list.append(accuracy_score(y_binary[test_idx], clf_rf.predict(X_te)))\n",
    "    lr_accs.append(np.mean(acc_lr_list) * 100)\n",
    "    rf_accs.append(np.mean(acc_rf_list) * 100)\n",
    "\n",
    "plt.figure(figsize=(11, 6))\n",
    "plt.plot(k_features, lr_accs, 'o-', color='#2980b9', lw=2.5, label='Logistic Regression 10-Fold Acc (%)')\n",
    "plt.plot(k_features, rf_accs, 's--', color='#e67e22', lw=2.5, label='Random Forest 10-Fold Acc (%)')\n",
    "peak_k = k_features[np.argmax(lr_accs)]\n",
    "peak_acc = np.max(lr_accs)\n",
    "plt.axvline(x=peak_k, color='red', linestyle=':', lw=2, label=f'Optimal Peak: k={peak_k} Features ({peak_acc:.2f}%)')\n",
    "plt.scatter([peak_k], [peak_acc], color='red', s=120, zorder=5)\n",
    "plt.title('Feature Selection Optimization Curve: Accuracy vs Number of Selected Features', fontsize=14, fontweight='bold')\n",
    "plt.xlabel('Number of Top Selected Landmark Features (k)', fontsize=12)\n",
    "plt.ylabel('10-Fold Cross-Validated Accuracy (%)', fontsize=12)\n",
    "plt.legend(loc='lower right', fontsize=11)\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

for idx, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "Evaluate Feature Selection Performance Curve" in src:
            cell["source"] = cell37_source
            cell["outputs"] = []
            cell["execution_count"] = None
            print(f"Successfully updated Cell index {idx} in notebook!")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Saved updated notebook.")
