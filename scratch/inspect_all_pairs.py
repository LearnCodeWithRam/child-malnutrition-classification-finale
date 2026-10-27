import pandas as pd
import numpy as np

excel_path = 'GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/copy-200-sample_mam-sam-onc_corrected_28-09-2026-finale.xlsx'
df = pd.read_excel(excel_path)

mam = df[df['Class_Type'] == 'mam'].copy()

# Check Y1 vs Y4 (shoulder), Y12 vs Y13, Y14 vs Y15, Y16 vs Y17, Y18 vs Y9, Y10 vs Y7, Y8 vs Y5, Y6 vs Y5, Y20 vs Y19
pairs = [
    (1, 4), (12, 13), (14, 15), (16, 17), (18, 9), (10, 7), (8, 5), (6, 5), (20, 19)
]

for top_id, bot_id in pairs:
    flipped = (mam[f'Y{top_id}'] < mam[f'Y{bot_id}']).sum()
    print(f"MAM pair Y{top_id} < Y{bot_id}: {flipped} / {len(mam)}")
