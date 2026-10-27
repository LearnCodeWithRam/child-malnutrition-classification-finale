import os
import json
import base64
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import StratifiedKFold

# Ensure output directory exists
os.makedirs('scratch', exist_ok=True)

# Path to notebook
nb_path = 'GM_out_of_sample_classification-ap-govt/SAM_MAM_ONC_Out_of_Sample_Classification.ipynb'
backup_path = 'backups/SAM_MAM_ONC_Out_of_Sample_Classification_backup.ipynb'

os.makedirs('backups', exist_ok=True)

# 1. Read notebook and create backup
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(backup_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print(f"Backup created at: {backup_path}")

# Load dataset
df_path = "GM_out_of_sample_classification-ap-govt/final_balanced_dataset_centred.csv"
df = pd.read_csv(df_path)

# Filter for SAM and ONC
df_2c = df[df['Class_Type'].isin(['sam', 'onc'])].copy().reset_index(drop=True)

x_cols = [c for c in df_2c.columns if c.startswith('X')]
y_cols = [c for c in df_2c.columns if c.startswith('Y')]
landmark_ids = [int(c[1:]) for c in x_cols]
k = len(landmark_ids)
n_2c = len(df_2c)

id_to_idx = {lid: idx for idx, lid in enumerate(landmark_ids)}

# 3D Landmark Tensor for 2-class
M_2c = np.zeros((k, 2, n_2c))
for i in range(n_2c):
    M_2c[:, 0, i] = df_2c.iloc[i][x_cols].values.astype(float)
    M_2c[:, 1, i] = df_2c.iloc[i][y_cols].values.astype(float)

labels_2c = df_2c['Class_Type'].values

# Core GPA functions
def center_shape(shape):
    return shape - np.mean(shape, axis=0)

def scale_shape(shape):
    cs = np.sqrt(np.sum(shape ** 2))
    return shape / cs, cs

def proc_opa(target, source):
    H = source.T @ target
    U, S, Vt = svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    aligned = source @ R
    return aligned

def proc_gpa(shapes, max_iterations=20, tolerance=1e-6):
    k, m, n_samples = shapes.shape
    aligned = np.zeros_like(shapes)
    for i in range(n_samples):
        centered = center_shape(shapes[:, :, i])
        scaled, _ = scale_shape(centered)
        aligned[:, :, i] = scaled
    mshape = aligned[:, :, 0].copy()
    for iteration in range(max_iterations):
        mshape_old = mshape.copy()
        for i in range(n_samples):
            aligned[:, :, i] = proc_opa(mshape, aligned[:, :, i])
        mshape = np.mean(aligned, axis=2)
        mshape, _ = scale_shape(mshape)
        if np.sum((mshape - mshape_old) ** 2) < tolerance:
            break
    return aligned, mshape

# Helper to run code and capture outputs
new_cells = []

def make_markdown_cell(source_str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_str.split("\n")]
    }

def make_code_cell(source_str, stdout_text=None, fig=None):
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_str.split("\n")]
    }
    if stdout_text:
        cell["outputs"].append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in stdout_text.split("\n")]
        })
    if fig is not None:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        cell["outputs"].append({
            "data": {
                "image/png": img_b64,
                "text/plain": "<Figure size 1200x800 with 1 Axes>"
            },
            "metadata": {},
            "output_type": "display_data"
        })
    return cell

print("Script template ready.")
