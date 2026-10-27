import os
import sys
import numpy as np
import pandas as pd
from scipy.linalg import svd, inv
from scipy.spatial.distance import cdist
import scipy.stats as stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import MDS
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score

print("Loading dataset...")
csv_path = r'e:/ap-govt-EDA-model-tainning-testing/GM_out_of_sample_classification-main/SAM-Photo/final_cleaned_dataset_full_with_muac.csv'
df_raw = pd.read_csv(csv_path)

# Clean muac_cm
muac_clean = np.where(df_raw['muac_cm'] > 40, df_raw['muac_cm'] / 10.0, df_raw['muac_cm'])
df_raw['muac_cm_clean'] = np.where(muac_clean < 5, np.nan, muac_clean)

# Step 1: Filter SAM vs ONC
df_eligible = df_raw[df_raw['Class_Type'].isin(['sam', 'onc'])].copy()
df_eligible['age_month'] = df_eligible['age_month'].clip(lower=6.0, upper=59.0)
df_eligible['gender'] = df_eligible['gender'].str.capitalize()

# Construct balanced cohort: 202 SAM and 208 ONC (N=410)
RANDOM_STATE = 42
sam_all = df_eligible[df_eligible['Class_Type'] == 'sam'].copy()
sam_cohort = sam_all.sample(n=202, random_state=RANDOM_STATE) if len(sam_all) >= 202 else sam_all

onc_all = df_eligible[df_eligible['Class_Type'] == 'onc'].copy()
target_rec = 202 - (sam_cohort['age_month'] <= 24.0).sum()
target_sta = 208 - (sam_cohort['age_month'] > 24.0).sum()
onc_rec = onc_all[onc_all['age_month'] <= 24.0].sample(n=int(target_rec), random_state=RANDOM_STATE)
onc_sta = onc_all[onc_all['age_month'] > 24.0].sample(n=int(target_sta), random_state=RANDOM_STATE)
onc_cohort = pd.concat([onc_rec, onc_sta])

df_cohort = pd.concat([sam_cohort, onc_cohort]).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
n_cohort = len(df_cohort)
print(f"Cohort constructed: Total {n_cohort} (SAM={sum(df_cohort['Class_Type']=='sam')}, ONC={sum(df_cohort['Class_Type']=='onc')})")

# Step 2: Coordinates extraction (20 landmarks)
# 19 raw landmarks: 1, 4, 12, 14, 16, 18, 10, 8, 6, 20, 13, 15, 17, 9, 7, 5, 19, 2 (Elbow), 3 (Wrist)
# Reconstruct Landmark 11 as forearm midpoint
df_cohort['X11'] = (df_cohort['X10'] + df_cohort['X12']) / 2.0
df_cohort['Y11'] = (df_cohort['Y10'] + df_cohort['Y12']) / 2.0

landmark_ids_20 = list(range(1, 21))
shapes_20 = np.zeros((20, 2, n_cohort))
for i, lid in enumerate(landmark_ids_20):
    shapes_20[i, 0, :] = df_cohort[f'X{lid}'].values
    shapes_20[i, 1, :] = df_cohort[f'Y{lid}'].values

def center_shape(shape):
    return shape - np.mean(shape, axis=0)

def scale_shape(shape):
    cs = np.sqrt(np.sum(shape ** 2))
    return (shape / cs if cs > 0 else shape), cs

def proc_opa(target, source):
    H = source.T @ target
    U, Sigma, Vt = svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return source @ R, R

def proc_gpa(shapes, max_iter=30, tol=1e-7):
    k, m, n = shapes.shape
    aligned = np.zeros_like(shapes)
    for i in range(n):
        c = center_shape(shapes[:, :, i])
        s, _ = scale_shape(c)
        aligned[:, :, i] = s
    mshape = aligned[:, :, 0].copy()
    for it in range(max_iter):
        mshape_old = mshape.copy()
        for i in range(n):
            aligned[:, :, i], _ = proc_opa(mshape, aligned[:, :, i])
        mshape = np.mean(aligned, axis=2)
        mshape, _ = scale_shape(mshape)
        if np.sum((mshape - mshape_old)**2) < tol:
            break
    return aligned, mshape

# Centroid sizes
centroid_sizes = np.array([scale_shape(center_shape(shapes_20[:, :, i]))[1] for i in range(n_cohort)])
log_cs = np.log(centroid_sizes)

shapes_gpa, mean_gpa = proc_gpa(shapes_20)

# Flatten coordinates: (n, 40)
X_40 = np.zeros((n_cohort, 40))
for i in range(n_cohort):
    X_40[i, :] = shapes_gpa[:, :, i].T.flatten()

# Allometric regression on log(CS)
reg = LinearRegression().fit(log_cs.reshape(-1, 1), X_40)
res_40 = X_40 - reg.predict(log_cs.reshape(-1, 1))

# 18 landmarks: drop landmarks 2 and 3 (indices 1 and 2 in 0-indexed list)
id_map = {lid: idx for idx, lid in enumerate(landmark_ids_20)}
land_keep_18 = [lid for lid in landmark_ids_20 if lid not in [2, 3]]
keep_indices_40 = []
for lid in land_keep_18:
    idx = id_map[lid]
    keep_indices_40.extend([idx, idx + 20])

X_36 = X_40[:, keep_indices_40]
res_36 = res_40[:, keep_indices_40]

cov_40 = np.cov(res_40, rowvar=False)
cov_36 = np.cov(res_36, rowvar=False)
eig_40 = np.linalg.eigvalsh(cov_40)
eig_36 = np.linalg.eigvalsh(cov_36)
cond_40 = eig_40[-1] / max(eig_40[0], 1e-18)
cond_36 = eig_36[-1] / max(eig_36[0], 1e-18)

print(f"Condition Numbers: 20-Landmark = {cond_40:.2e}, 18-Landmark = {cond_36:.2e}")

# Goodall F-statistic
def goodall_f(coords, labels, k):
    n = len(labels)
    unique_labels = np.unique(labels)
    g = len(unique_labels)
    grand_mean = np.mean(coords, axis=0)
    ss_total = np.sum((coords - grand_mean)**2)
    ss_grp = 0.0
    for grp in unique_labels:
        idx = np.where(labels == grp)[0]
        grp_mean = np.mean(coords[idx, :], axis=0)
        ss_grp += len(idx) * np.sum((grp_mean - grand_mean)**2)
    ss_res = ss_total - ss_grp
    df1 = (g - 1) * (2 * k - 4)
    df2 = (n - g) * (2 * k - 4)
    ms_grp = ss_grp / (g - 1)
    ms_res = ss_res / (n - g)
    F = ms_grp / ms_res
    p = 1.0 - stats.f.cdf(F, df1, df2)
    return F, p

y_status = df_cohort['Class_Type'].values
y_age = (df_cohort['age_month'] <= 24).astype(str).values
y_sex = df_cohort['gender'].values

print("\n--- PROCRUSTES ANOVA (GOODALL F) RESULTS ON DATASET ---")
F_nutr_raw, p_nutr_raw = goodall_f(X_36, y_status, 18)
F_nutr_res, p_nutr_res = goodall_f(res_36, y_status, 18)
print(f"Nutritional Status: Before F={F_nutr_raw:.2f} (p={p_nutr_raw:.4e}) -> After F={F_nutr_res:.2f} (p={p_nutr_res:.4e})")

F_age_raw, p_age_raw = goodall_f(X_36, y_age, 18)
F_age_res, p_age_res = goodall_f(res_36, y_age, 18)
print(f"Age Group:          Before F={F_age_raw:.2f} (p={p_age_raw:.4e}) -> After F={F_age_res:.2f} (p={p_age_res:.4e})")

F_sex_raw, p_sex_raw = goodall_f(X_36, y_sex, 18)
F_sex_res, p_sex_res = goodall_f(res_36, y_sex, 18)
print(f"Sex:                Before F={F_sex_raw:.2f} (p={p_sex_raw:.4e}) -> After F={F_sex_res:.2f} (p={p_sex_res:.4e})")

# Authentic LOOCV Evaluation
y_true = np.array([1 if c == 'sam' else 0 for c in y_status])
print("\n--- RUNNING OUT-OF-SAMPLE LOOCV EXPERIMENTS ---")
pred_lda_mean = np.zeros(n_cohort)
pred_lda_med  = np.zeros(n_cohort)
pred_knn_mean = np.zeros(n_cohort)
pred_knn_med  = np.zeros(n_cohort)
pred_lr_med   = np.zeros(n_cohort)
pred_lr_mean  = np.zeros(n_cohort)
pred_lda_unc  = np.zeros(n_cohort)
pred_lda_col  = np.zeros(n_cohort)

# 18-landmark raw shapes
idx_18 = [id_map[lid] for lid in land_keep_18]
shapes_18 = shapes_20[idx_18, :, :]
mean_18 = mean_gpa[idx_18, :]
# functional median
dists = np.sum((shapes_gpa - mean_gpa[:, :, np.newaxis])**2, axis=(0, 1))
med_18 = shapes_gpa[idx_18, :, np.argmin(dists)]


for i in range(n_cohort):
    train_idx = [j for j in range(n_cohort) if j != i]
    X_tr_18 = res_36[train_idx, :]
    y_tr = y_true[train_idx]
    
    # Test shape alignment (fOPA to mean and median)
    t_shp = shapes_18[:, :, i]
    t_c = center_shape(t_shp)
    t_s, _ = scale_shape(t_c)
    t_mean, _ = proc_opa(mean_18, t_s)
    t_med, _ = proc_opa(med_18, t_s)
    
    # Test residual
    t_res_mean = (t_mean.T.flatten() - reg.intercept_[keep_indices_40] - reg.coef_[keep_indices_40].flatten() * log_cs[i]).reshape(1, -1)
    t_res_med  = (t_med.T.flatten()  - reg.intercept_[keep_indices_40] - reg.coef_[keep_indices_40].flatten() * log_cs[i]).reshape(1, -1)
    
    # 1. Optimal LDA (Mean)
    clf_lda = LinearDiscriminantAnalysis().fit(X_tr_18, y_tr)
    pred_lda_mean[i] = clf_lda.predict(t_res_mean)[0]
    pred_lda_med[i]  = clf_lda.predict(t_res_med)[0]
    
    # 2. kNN (k=5)
    clf_knn = KNeighborsClassifier(n_neighbors=5).fit(X_tr_18, y_tr)
    pred_knn_mean[i] = clf_knn.predict(t_res_mean)[0]
    pred_knn_med[i]  = clf_knn.predict(t_res_med)[0]
    
    # 3. Logistic Regression
    clf_lr = LogisticRegression(max_iter=500).fit(X_tr_18, y_tr)
    pred_lr_med[i]  = clf_lr.predict(t_res_med)[0]
    pred_lr_mean[i] = clf_lr.predict(t_res_mean)[0]
    
    # 4. LDA Uncorrected
    clf_unc = LinearDiscriminantAnalysis().fit(X_36[train_idx, :], y_tr)
    pred_lda_unc[i] = clf_unc.predict(t_mean.T.flatten().reshape(1, -1))[0]
    
    # 5. LDA Collinear (20 landmarks, singular)
    clf_col = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=1e-8).fit(res_40[train_idx, :], y_tr)
    pred_lda_col[i] = clf_col.predict(res_40[i, :].reshape(1, -1))[0]

def get_metrics(y_t, y_p):
    acc = accuracy_score(y_t, y_p) * 100
    sens = recall_score(y_t, y_p, pos_label=1) * 100
    spec = recall_score(y_t, y_p, pos_label=0) * 100
    return acc, sens, spec

print("\n=== AUTHENTIC MODEL METRICS FROM DATASET EXPERIMENTS ===")
models = {
    'LDA (Optimal / Final)': pred_lda_mean,
    'LDA (Functional Median)': pred_lda_med,
    'kNN (k=5, Mean Target)': pred_knn_mean,
    'kNN (k=5, Median Target)': pred_knn_med,
    'Logistic Regression (Median)': pred_lr_med,
    'Logistic Regression (Mean)': pred_lr_mean,
    'LDA (Uncorrected)': pred_lda_unc,
    'LDA (Collinear 20 Landmarks)': pred_lda_col
}

for name, preds in models.items():
    acc, sens, spec = get_metrics(y_true, preds)
    print(f"{name:32s}: Accuracy={acc:6.2f}% | Sensitivity={sens:6.2f}% | Specificity={spec:6.2f}%")
