import pandas as pd
import numpy as np

excel_path = 'GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/copy-200-sample_mam-sam-onc_corrected_28-09-2026-finale.xlsx'
df = pd.read_excel(excel_path)

for cls in ['sam', 'mam', 'onc']:
    sub = df[df['Class_Type'] == cls]
    print(f"=== Class {cls.upper()} (n={len(sub)}) ===")
    for lid in [5, 6, 19, 20, 3]:
        print(f"L{lid:2d} -> X: min={sub[f'X{lid}'].min():.2f}, mean={sub[f'X{lid}'].mean():.2f}, max={sub[f'X{lid}'].max():.2f} | Y: mean={sub[f'Y{lid}'].mean():.2f}")
