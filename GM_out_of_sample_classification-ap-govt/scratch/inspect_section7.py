import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/SAM_MAM_ONC_Out_of_Sample_Classification.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if idx >= 33:
        ctype = cell['cell_type']
        print(f"=== Cell {idx} ({ctype}) ===")
        print(src)
        print("\n" + "="*60 + "\n")
