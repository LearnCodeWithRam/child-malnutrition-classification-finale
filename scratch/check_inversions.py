import pandas as pd
import numpy as np

excel_path = 'GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/copy-200-sample_mam-sam-onc_corrected_28-09-2026-finale.xlsx'
df = pd.read_excel(excel_path)

mam = df[df['Class_Type'] == 'mam'].copy()

# Check Y20 vs Y19 in MAM: in normal arm orientation, Y20 (upper wrist) > Y19 (lower wrist)
inverted = mam[mam['Y20'] < mam['Y19']]
print(f"MAM samples with Y20 < Y19 (vertical inversion): {len(inverted)} out of {len(mam)}")

# Check X19 vs X5 in MAM: in normal arm, X19 (wrist) > X5 (forearm)
x19_inverted = mam[mam['X19'] < mam['X5']]
print(f"MAM samples with X19 < X5 (horizontal inversion/misplacement): {len(x19_inverted)} out of {len(mam)}")

# Check SAM and ONC for the same
sam = df[df['Class_Type'] == 'sam']
onc = df[df['Class_Type'] == 'onc']
print(f"SAM Y20 < Y19: {(sam['Y20'] < sam['Y19']).sum()} | SAM X19 < X5: {(sam['X19'] < sam['X5']).sum()}")
print(f"ONC Y20 < Y19: {(onc['Y20'] < onc['Y19']).sum()} | ONC X19 < X5: {(onc['X19'] < onc['X5']).sum()}")
