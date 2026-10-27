import matplotlib
matplotlib.use('Agg')
import json
import traceback
import sys

path = r'e:\ap-govt-EDA-model-tainning-testing\GM_out_of_sample_classification-main\GM_out_of_sample_classification-ap-govt\SAM_MAM_ONC_Out_of_Sample_Classification.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

globals_dict = {}

for i in range(len(nb['cells'])):
    c = nb['cells'][i]
    if c['cell_type'] == 'code':
        src = ''.join(c['source'])
        print(f"=== Executing Cell {i} ===")
        try:
            exec(src, globals_dict)
            print(f"Cell {i} executed successfully.")
        except Exception as e:
            print(f"Error in Cell {i}: {e}")
            traceback.print_exc()
            break
