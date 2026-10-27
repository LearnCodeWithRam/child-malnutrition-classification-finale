import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

path = r'GM_out_of_sample_classification-main/GM_out_of_sample_classification-ap-govt/SAM_MAM_ONC_Out_of_Sample_Classification.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

globals_dict = {}

# Execute cells 1, 3, 5, 7, 8, 10
for idx in [1, 3, 5, 7, 8, 10]:
    exec(''.join(nb['cells'][idx]['source']), globals_dict)

aligned_shapes_rot = globals_dict['aligned_shapes_rot']
true_class = globals_dict['true_class']
landmark_ids = globals_dict['landmark_ids']
id_to_idx = globals_dict['id_to_idx']
plot_arm_wireframe = globals_dict['plot_arm_wireframe']

sam_indices = np.where(true_class == 'sam')[0]
onc_indices = np.where(true_class == 'onc')[0]

sam_mean = np.mean(aligned_shapes_rot[:, :, sam_indices], axis=2)
onc_mean = np.mean(aligned_shapes_rot[:, :, onc_indices], axis=2)

print("sam_mean shape:", sam_mean.shape)
print("onc_mean shape:", onc_mean.shape)

# Transverse top-down cross-section pairs along the arm from shoulder to wrist:
transverse_pairs = [
    (1, 4),    # Shoulder edge
    (12, 4),   # Upper arm slice 1
    (14, 13),  # Upper arm slice 2
    (16, 15),  # Mid-arm slice
    (18, 17),  # Elbow slice
    (10, 9),   # Forearm slice 1
    (8, 7),    # Forearm slice 2
    (6, 5),    # Forearm slice 3
    (20, 19)   # Wrist edge
]

print("Transverse pairs defined:", transverse_pairs)
