import pandas as pd
import numpy as np

excel_path = 'GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/copy-200-sample_mam-sam-onc_corrected_28-09-2026-finale.xlsx'
df = pd.read_excel(excel_path)

mam = df[df['Class_Type'] == 'mam']

print("--- MAM X19 Values ---")
print(mam['X19'].describe())

print("\n--- MAM X20 Values ---")
print(mam['X20'].describe())

print("\n--- MAM Y20 Values ---")
print(mam['Y20'].describe())

# Check how many samples in MAM have X19 < 25 vs >= 25
print("\nMAM samples with X19 < 25:", (mam['X19'] < 25).sum(), "out of", len(mam))
print("MAM samples with Y20 < 1.0:", (mam['Y20'] < 1.0).sum(), "out of", len(mam))

# Compare with SAM and ONC
sam = df[df['Class_Type'] == 'sam']
onc = df[df['Class_Type'] == 'onc']
print("\nSAM samples with X19 < 25:", (sam['X19'] < 25).sum(), "out of", len(sam))
print("ONC samples with X19 < 25:", (onc['X19'] < 25).sum(), "out of", len(onc))
