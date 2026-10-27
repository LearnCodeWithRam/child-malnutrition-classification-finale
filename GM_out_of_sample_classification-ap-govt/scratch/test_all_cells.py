import matplotlib
matplotlib.use('Agg')
import json
import traceback

path = r'GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/SAM_MAM_ONC_Out_of_Sample_Classification.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

globals_dict = {}

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell.get('source', []))
        print(f"=== Testing Cell {idx} ===")
        try:
            exec(src, globals_dict)
            print(f"Cell {idx} PASS")
        except Exception as e:
            print(f"Cell {idx} FAIL: {e}")
            traceback.print_exc()
