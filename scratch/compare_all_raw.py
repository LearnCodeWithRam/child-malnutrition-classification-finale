import pandas as pd

excel_path = 'GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/copy-200-sample_mam-sam-onc_corrected_28-09-2026-finale.xlsx'
df = pd.read_excel(excel_path)

landmark_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20]

sam = df[df['Class_Type'] == 'sam']
mam = df[df['Class_Type'] == 'mam']
onc = df[df['Class_Type'] == 'onc']

print(f"{'ID':<4} | {'SAM X':<7} {'SAM Y':<7} | {'MAM X':<7} {'MAM Y':<7} | {'ONC X':<7} {'ONC Y':<7}")
print("-" * 55)

for lid in landmark_ids:
    sx, sy = sam[f'X{lid}'].mean(), sam[f'Y{lid}'].mean()
    mx, my = mam[f'X{lid}'].mean(), mam[f'Y{lid}'].mean()
    ox, oy = onc[f'X{lid}'].mean(), onc[f'Y{lid}'].mean()
    print(f"{lid:<4} | {sx:<7.2f} {sy:<7.2f} | {mx:<7.2f} {my:<7.2f} | {ox:<7.2f} {oy:<7.2f}")
