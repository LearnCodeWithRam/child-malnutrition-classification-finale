import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os

path = r'e:\ap-govt-EDA-model-tainning-testing\GM_out_of_sample_classification-main\GM_out_of_sample_classification-ap-govt\SAM_MAM_ONC_Out_of_Sample_Classification.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

globals_dict = {}

# Execute cells 1, 3, 5, 7, 8, 9
for i in [1, 3, 5, 7, 8, 9]:
    c = nb['cells'][i]
    src = ''.join(c['source'])
    exec(src, globals_dict)

# Save current figure
plt.savefig(r'e:\ap-govt-EDA-model-tainning-testing\GM_out_of_sample_classification-main\GM_out_of_sample_classification-ap-govt\scratch\step3_plot.png', bbox_inches='tight', dpi=150)
print("Saved step3_plot.png successfully!")
