import matplotlib
matplotlib.use('Agg')
import json

path = r'GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/SAM_MAM_ONC_Out_of_Sample_Classification.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

globals_dict = {}

# Execute cells 1, 3, 5, 23, 24, 25, 26, 27, 28, 31, 32, 35, 36, 37, 38, 41, 42, 43, 44
test_cell_indices = [1, 3, 5, 23, 24, 25, 26, 27, 28, 31, 32, 35, 36, 37, 38, 41, 42, 43, 44]

for idx in test_cell_indices:
    cell = nb['cells'][idx]
    src = ''.join(cell['source'])
    print(f'Testing cell {idx}...')
    try:
        exec(src, globals_dict)
        print(f'Cell {idx} PASS')
    except Exception as e:
        print(f'Cell {idx} FAIL: {e}')
        raise e

print("ALL TESTED CELLS (23 TO 44) PASSED WITH 0 ERRORS!")
