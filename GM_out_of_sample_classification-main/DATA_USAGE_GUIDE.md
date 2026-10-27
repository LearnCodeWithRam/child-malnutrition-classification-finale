# 📊 Data Usage Guide

## Current Status: Ready for Real Gorilla Data!

The Jupyter notebook is now configured to work with **real gorilla skull data** from the R `shapes` package, with automatic fallback to sample data.

---

## 🦍 Using Real Gorilla Data

### Step 1: Export from R

Run this in R (or use the provided script):

```r
source("export_data_from_R.R")
```

This creates two files:
- `gorf_data.npy` - Female gorilla skulls (30 individuals, 8 landmarks)
- `gorm_data.npy` - Male gorilla skulls (29 individuals, 8 landmarks)

### Step 2: Run the Notebook

```bash
jupyter notebook GM_OUT_Geometric_Classification.ipynb
```

The notebook will automatically:
1. ✅ Try to load `gorf_data.npy` and `gorm_data.npy`
2. ✅ If found: Use real data
3. ⚠️ If not found: Use sample data (still demonstrates the method)

---

## 💪 Switching to Arm Data (When Available)

When you receive arm contour data for SAM detection, follow these steps:

### Step 1: Prepare Your Arm Data

Your arm data should be in the same format:
- **Shape**: `(k, 2, n)` where:
  - `k` = number of landmarks (e.g., 8)
  - `2` = coordinates (x, y)
  - `n` = number of individuals

Example:
```python
import numpy as np

# Healthy arms: 8 landmarks, 2D, 50 individuals
healthy_arms = np.zeros((8, 2, 50))
# ... fill with your landmark data ...

# SAM arms: 8 landmarks, 2D, 50 individuals  
sam_arms = np.zeros((8, 2, 50))
# ... fill with your landmark data ...

# Save as .npy files
np.save("healthy_arms.npy", healthy_arms)
np.save("sam_arms.npy", sam_arms)
```

### Step 2: Update the Notebook

In the data loading cell, change:

```python
# OLD (gorilla data):
sk_F = np.load("gorf_data.npy")
sk_M = np.load("gorm_data.npy")

# NEW (arm data):
sk_F = np.load("healthy_arms.npy")
sk_M = np.load("sam_arms.npy")
```

### Step 3: Update Labels

Change the labels to reflect your data:

```python
# OLD:
true_class = np.array(['Female'] * n1 + ['Male'] * n2)

# NEW:
true_class = np.array(['Healthy'] * n1 + ['SAM'] * n2)
```

### Step 4: Run!

That's it! The entire Procrustes analysis and classification pipeline works exactly the same way.

---

## 📐 Data Format Requirements

### Input Format

Your data must be a 3D NumPy array:

```
Shape: (k, 2, n)
  k = number of landmarks per individual
  2 = x and y coordinates
  n = number of individuals
```

### Example Structure

```python
# For one individual with 8 landmarks:
individual_1 = np.array([
    [x1, y1],  # Landmark 1
    [x2, y2],  # Landmark 2
    [x3, y3],  # Landmark 3
    [x4, y4],  # Landmark 4
    [x5, y5],  # Landmark 5
    [x6, y6],  # Landmark 6
    [x7, y7],  # Landmark 7
    [x8, y8]   # Landmark 8
])

# For multiple individuals:
all_individuals = np.zeros((8, 2, n))
all_individuals[:, :, 0] = individual_1
all_individuals[:, :, 1] = individual_2
# ... and so on
```

---

## 🔄 Data Conversion Examples

### From CSV

```python
import pandas as pd
import numpy as np

# Assuming CSV format: individual_id, landmark_id, x, y
df = pd.read_csv("arm_landmarks.csv")

k = df['landmark_id'].max()
n = df['individual_id'].max()

data = np.zeros((k, 2, n))
for i in range(n):
    for j in range(k):
        row = df[(df['individual_id'] == i+1) & (df['landmark_id'] == j+1)]
        data[j, 0, i] = row['x'].values[0]
        data[j, 1, i] = row['y'].values[0]

np.save("arm_data.npy", data)
```

### From Images (with landmark detection)

```python
# Pseudo-code - adapt to your landmark detection method
import cv2
import numpy as np

def extract_landmarks(image_path):
    # Your landmark detection code here
    # Returns: array of shape (k, 2)
    pass

# Process all images
n_images = 100
k_landmarks = 8
all_landmarks = np.zeros((k_landmarks, 2, n_images))

for i, image_path in enumerate(image_paths):
    landmarks = extract_landmarks(image_path)
    all_landmarks[:, :, i] = landmarks

np.save("extracted_landmarks.npy", all_landmarks)
```

---

## ✅ Verification

After loading your data, verify the format:

```python
print(f"Shape: {data.shape}")
print(f"Expected: (k_landmarks, 2, n_individuals)")
print(f"Landmarks: {data.shape[0]}")
print(f"Coordinates: {data.shape[1]} (should be 2)")
print(f"Individuals: {data.shape[2]}")

# Check for NaN or inf values
print(f"Has NaN: {np.isnan(data).any()}")
print(f"Has inf: {np.isinf(data).any()}")
```

---

## 🎯 Summary

| Data Type | File Names | Labels | Status |
|-----------|------------|--------|--------|
| **Gorilla Skulls** | `gorf_data.npy`, `gorm_data.npy` | Female, Male | ✅ Ready Now |
| **Arm Contours** | `healthy_arms.npy`, `sam_arms.npy` | Healthy, SAM | 🔄 When Available |

**The methodology is identical - only the data source changes!**

---

## 📞 Need Help?

- Check `NOTEBOOK_GUIDE.md` for notebook usage
- Check `README_PYTHON.md` for technical details
- Check `HOW_TO_RUN.md` for setup instructions

The geometric morphometrics pipeline is universal and works for any landmark-based shape analysis! 🎉
