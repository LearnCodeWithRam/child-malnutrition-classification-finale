# -*- coding: utf-8 -*-
"""
Full Automated Notebook Generator for SAM Photo Diagnosis Methodology Pipeline
Builds: SAM_Photo_Diagnosis_Methodology_Pipeline.ipynb
"""
import os
import sys
import io
import json
import base64
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy.linalg import svd, inv
from scipy.spatial.distance import cdist
import scipy.stats as stats

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import MDS
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_curve, auc

print("Initializing Notebook Generator for SAM Photo Diagnosis Methodology...")

notebook_dir = r"e:\ap-govt-EDA-model-tainning-testing\GM_out_of_sample_classification-main\SAM-Photo"
notebook_path = os.path.join(notebook_dir, "SAM_Photo_Diagnosis_Methodology_Pipeline.ipynb")

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code_and_run(code_str, env_globals):
    buf_stdout = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf_stdout
    
    plt.close('all')
    outputs = []
    error_occurred = False
    
    try:
        exec(code_str, env_globals)
    except Exception as e:
        error_occurred = True
        print(f"Error executing cell: {e}")
        traceback.print_exc()
    finally:
        sys.stdout = old_stdout
    
    stdout_text = buf_stdout.getvalue()
    if stdout_text:
        outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": stdout_text.splitlines(True)
        })
        
    figs = [plt.figure(i) for i in plt.get_fignums()]
    for fig in figs:
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=120)
        img_buf.seek(0)
        img_b64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
        outputs.append({
            "data": {
                "image/png": img_b64,
                "text/plain": "<Figure size ...>"
            },
            "metadata": {},
            "output_type": "display_data"
        })
        plt.close(fig)
        
    cells.append({
        "cell_type": "code",
        "execution_count": len([c for c in cells if c['cell_type'] == 'code']) + 1,
        "metadata": {},
        "outputs": outputs,
        "source": [line + "\n" for line in code_str.strip().split("\n")]
    })
    
    if error_occurred:
        print(f"FAILED on cell execution! Output:\n{stdout_text}")
        sys.exit(1)

env = {}
print("Generator framework ready. Adding cells...")
