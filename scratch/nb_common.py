# -*- coding: utf-8 -*-
"""
Common utilities for building the SAM Photo Diagnosis Methodology Pipeline Notebook
"""
import os
import sys
import io
import json
import base64
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

notebook_dir = r"e:\ap-govt-EDA-model-tainning-testing\GM_out_of_sample_classification-main\SAM-Photo"
notebook_path = os.path.join(notebook_dir, "SAM_Photo_Diagnosis_Methodology_Pipeline.ipynb")

cells = []
env = {'notebook_dir': notebook_dir}


def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code_and_run(code_str):
    buf_stdout = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf_stdout
    
    plt.close('all')
    outputs = []
    error_occurred = False
    
    try:
        exec(code_str, env)
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

def save_notebook():
    os.makedirs(notebook_dir, exist_ok=True)
    nb_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=1, ensure_ascii=False)
    print(f"\n[SUCCESS] Notebook saved to: {notebook_path}")
    print(f"Total cells: {len(cells)} (Code: {len([c for c in cells if c['cell_type'] == 'code'])}, Markdown: {len([c for c in cells if c['cell_type'] == 'markdown'])})")
