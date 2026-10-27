import pandas as pd
import numpy as np

excel_path = 'GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/copy-200-sample_mam-sam-onc_corrected_28-09-2026-finale.xlsx'
df = pd.read_excel(excel_path)

landmark_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20]
id_to_idx = {lid: i for i, lid in enumerate(landmark_ids)}

# Let's apply the vertical flip fix requested by user:
# Fix vertical flips where Y20 < Y19
flip_mask = df['Y20'] < df['Y19']
df_fixed = df.copy()

# Swap X20, X19 and Y20, Y19
df_fixed.loc[flip_mask, ['X20', 'X19']] = df_fixed.loc[flip_mask, ['X19', 'X20']].values
df_fixed.loc[flip_mask, ['Y20', 'Y19']] = df_fixed.loc[flip_mask, ['Y19', 'Y20']].values

# Also let's test swapping all upper/lower contour pairs that were inverted in those samples:
contour_pairs = [(12, 13), (14, 15), (16, 17), (18, 9), (20, 19)]
for u_id, l_id in contour_pairs:
    pair_mask = df_fixed[f'Y{u_id}'] < df_fixed[f'Y{l_id}']
    df_fixed.loc[pair_mask, [f'X{u_id}', f'X{l_id}']] = df_fixed.loc[pair_mask, [f'X{l_id}', f'X{u_id}']].values
    df_fixed.loc[pair_mask, [f'Y{u_id}', f'Y{l_id}']] = df_fixed.loc[pair_mask, [f'Y{l_id}', f'Y{u_id}']].values

x_cols = [f'X{i}' for i in landmark_ids]
y_cols = [f'Y{i}' for i in landmark_ids]

X_pts = df_fixed[x_cols].values
Y_pts = df_fixed[y_cols].values
n_samples = len(df_fixed)
n_landmarks = len(landmark_ids)

M = np.zeros((n_landmarks, 2, n_samples))
for i in range(n_samples):
    M[:, 0, i] = X_pts[i, :]
    M[:, 1, i] = Y_pts[i, :]

def center_shape(shape):
    centroid = np.mean(shape, axis=0)
    return shape - centroid, centroid

def scale_shape(shape):
    centroid_size = np.sqrt(np.sum(shape ** 2))
    return shape / centroid_size, centroid_size

def align_shapes_procrustes(source, target):
    u, s, vt = np.linalg.svd(np.dot(target.T, source))
    R = np.dot(vt.T, u.T)
    if np.linalg.det(R) < 0:
        vt[1, :] *= -1
        R = np.dot(vt.T, u.T)
    return np.dot(source, R)

def proc_gpa(shapes, max_iter=100, tol=1e-7):
    k, d, n = shapes.shape
    centered_scaled = np.zeros_like(shapes)
    for i in range(n):
        c_shp, _ = center_shape(shapes[:, :, i])
        s_shp, _ = scale_shape(c_shp)
        centered_scaled[:, :, i] = s_shp
    consensus = centered_scaled[:, :, 0].copy()
    for iteration in range(max_iter):
        aligned = np.zeros_like(centered_scaled)
        for i in range(n):
            aligned[:, :, i] = align_shapes_procrustes(centered_scaled[:, :, i], consensus)
        new_consensus = np.mean(aligned, axis=2)
        new_consensus, _ = center_shape(new_consensus)
        new_consensus, _ = scale_shape(new_consensus)
        diff = np.linalg.norm(new_consensus - consensus)
        consensus = new_consensus.copy()
        if diff < tol:
            break
    return {'mshape': consensus, 'rotated': aligned}

gpa_all = proc_gpa(M)
aligned_shapes = gpa_all['rotated']
mean_shape_overall = gpa_all['mshape']

ref_vector = mean_shape_overall[id_to_idx[3], :] - mean_shape_overall[id_to_idx[1], :]
angle = -np.arctan2(ref_vector[1], ref_vector[0])
R_horiz = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

aligned_shapes_rot = np.zeros_like(aligned_shapes)
for i in range(n_samples):
    aligned_shapes_rot[:, :, i] = np.dot(aligned_shapes[:, :, i], R_horiz)

true_class = df_fixed['Class_Type'].values

for cls in ['sam', 'mam', 'onc']:
    idx = np.where(true_class == cls)[0]
    m_cls = np.mean(aligned_shapes_rot[:, :, idx], axis=2)
    print(f"\n=== FIXED GPA Mean for {cls.upper()} ===")
    for lid in [1, 2, 3, 19, 20]:
        print(f"Landmark {lid:2d} (idx {id_to_idx[lid]:2d}): X={m_cls[id_to_idx[lid], 0]:7.4f}, Y={m_cls[id_to_idx[lid], 1]:7.4f}")
