import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import json

path = r'GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/SAM_MAM_ONC_Out_of_Sample_Classification.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

globals_dict = {}

# Execute cells 1, 3, 5, 23
for idx in [1, 3, 5, 23]:
    exec(''.join(nb['cells'][idx]['source']), globals_dict)

df_2c = globals_dict['df_2c']
x_cols = globals_dict['x_cols']
y_cols = globals_dict['y_cols']
landmark_ids = globals_dict['landmark_ids']
y_binary = globals_dict['y_binary']

X_raw = df_2c[x_cols + y_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 1. Individual feature metrics
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y_binary)
rf_imp = rf.feature_importances_

lr_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, random_state=42)
lr_l1.fit(X_scaled, y_binary)
lasso_coefs = np.abs(lr_l1.coef_[0])

f_vals, p_vals = f_classif(X_scaled, y_binary)

# 2. Combine paired (Xi, Yi) metrics per landmark
num_lms = len(landmark_ids)
# x_cols are indices 0..18, y_cols are indices 19..37
l_rows = []
for i, lid in enumerate(landmark_ids):
    x_idx = i
    y_idx = i + num_lms
    
    lasso_pair = lasso_coefs[x_idx] + lasso_coefs[y_idx]
    rf_pair = rf_imp[x_idx] + rf_imp[y_idx]
    f_pair = f_vals[x_idx] + f_vals[y_idx]
    
    l_rows.append({
        'Landmark_ID': f'L{lid} (X{lid}, Y{lid})',
        'Raw_ID': lid,
        'Combined_Lasso_L1': lasso_pair,
        'Combined_RF_Importance': rf_pair,
        'Combined_ANOVA_F': f_pair
    })

df_lm_imp = pd.DataFrame(l_rows).sort_values(by='Combined_Lasso_L1', ascending=False).reset_index(drop=True)
print("=== PAIRED 2D LANDMARK IMPORTANCE (SORTED BY COMBINED LASSO L1) ===")
print(df_lm_imp.to_string(index=False))

# 3. Test Feature Selection by Landmark count K (1..19)
sorted_lids = df_lm_imp['Raw_ID'].values
k_landmarks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 19]
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

lr_accs, rf_accs, svm_accs = [], [], []
for K in k_landmarks:
    top_k_lids = sorted_lids[:K]
    cols = []
    for lid in top_k_lids:
        cols.append(f'X{lid}')
        cols.append(f'Y{lid}')
    X_k = df_2c[cols].values
    
    acc_lr, acc_rf, acc_svm = [], [], []
    for train_idx, test_idx in skf.split(X_k, y_binary):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_k[train_idx])
        X_te = sc.transform(X_k[test_idx])
        
        clf_lr = LogisticRegression(max_iter=1000, random_state=42)
        clf_lr.fit(X_tr, y_binary[train_idx])
        acc_lr.append(accuracy_score(y_binary[test_idx], clf_lr.predict(X_te)))
        
        clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_rf.fit(X_tr, y_binary[train_idx])
        acc_rf.append(accuracy_score(y_binary[test_idx], clf_rf.predict(X_te)))
        
        clf_svm = SVC(kernel='rbf', random_state=42)
        clf_svm.fit(X_tr, y_binary[train_idx])
        acc_svm.append(accuracy_score(y_binary[test_idx], clf_svm.predict(X_te)))
        
    lr_accs.append(np.mean(acc_lr) * 100)
    rf_accs.append(np.mean(acc_rf) * 100)
    svm_accs.append(np.mean(acc_svm) * 100)

print("\n=== ACCURACY VS NUMBER OF SELECTED COMPLETE 2D LANDMARKS (K) ===")
df_curve = pd.DataFrame({
    'K_Landmarks': k_landmarks,
    'Num_Coords (2K)': [2*k for k in k_landmarks],
    'LR_Accuracy (%)': lr_accs,
    'RF_Accuracy (%)': rf_accs,
    'SVM_Accuracy (%)': svm_accs
})
print(df_curve.to_string(index=False))
