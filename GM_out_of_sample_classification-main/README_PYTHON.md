# Python Conversion of GM Out-of-Sample Classification

This directory contains Python conversions of the R code for "Classification of nutritional status of children from Geometric Morphometrics: an approach to out-of-sample data" (2024).

## Files Converted

1. **aux_functions.py** - Auxiliary functions (converted from aux_functions.R)
2. **code_out_of_sample_class.py** - Main classification code (converted from code_out_of_sample_class.R)
3. **Simulation_study.py** - Full simulation study (converted from Simulation_study.R)
4. **Simulation_study_example_visualization.py** - Simulation visualization (converted from Simulation_study_example_visualization.R)

## Dependencies

Install required Python packages:

```bash
pip install numpy scipy scikit-learn pandas matplotlib
```

### R Package Equivalents

The Python code replaces the following R packages:

- **shapes** → Custom implementations of `procGPA`, `procOPA`, `procWGPA`
- **MASS** → `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`
- **class** → `sklearn.neighbors.KNeighborsClassifier`
- **roahd** → Custom implementation of functional median
- **abind** → `numpy.concatenate`
- **geomorph** → Custom implementation of allometric regression
- **fdasrvf** → Not implemented (SRVF alignment functions commented out)

## Key Differences from R Code

1. **Array Indexing**: Python uses 0-based indexing (R uses 1-based)
2. **Array Dimensions**: NumPy arrays follow (rows, columns, depth) convention
3. **Data Structures**: Pandas DataFrames replace R data.frames
4. **Plotting**: Matplotlib replaces R's base graphics
5. **SRVF Functions**: Not implemented - requires separate fdasrvf Python package

## Data Requirements

The code requires gorilla skull data from the R 'shapes' package:
- Female skulls: `shapes::gorf.dat`
- Male skulls: `shapes::gorm.dat`

### Exporting Data from R

To use the Python code, export the data from R:

```r
library(shapes)

# Export as CSV
write.csv(gorf.dat, "gorf_data.csv")
write.csv(gorm.dat, "gorm_data.csv")

# Or export as NumPy arrays (requires reticulate)
library(reticulate)
np <- import("numpy")
np$save("gorf_data.npy", gorf.dat)
np$save("gorm_data.npy", gorm.dat)
```

Then load in Python:

```python
import numpy as np

# From NumPy files
sk_F = np.load("gorf_data.npy")
sk_M = np.load("gorm_data.npy")
```

## Usage

```python
import numpy as np
from aux_functions import proc_gpa, loo_cv_oos_classification

# Load your data (k landmarks, 2 dimensions, n individuals)
sk_F = np.load("gorf_data.npy")  # Shape: (8, 2, 30)
sk_M = np.load("gorm_data.npy")  # Shape: (8, 2, 29)

# Combine data
sk = np.concatenate([sk_F, sk_M], axis=2)
k, _, n = sk.shape
n1, n2 = sk_F.shape[2], sk_M.shape[2]

# Create class labels
true_class = np.array(['F'] * n1 + ['M'] * n2)

# Run classification
results = loo_cv_oos_classification(n, k, sk, true_class, all_ef=True, knnk=5)

# View results
import pandas as pd
print(pd.crosstab(results['predicted_class_LDA']['mean_registration'],
                  results['predicted_class_LDA']['true_class']))
```

## Notes

1. The SRVF alignment methods are not implemented as they require the `fdasrvf` Python package
2. Some plotting functions may need adjustment based on your specific visualization needs
3. The functional median computation uses a simplified coordinate-wise approach
4. Performance may differ slightly from R due to different optimization algorithms in scikit-learn

## Original R Code

The original R code is preserved in the `.R` files in this directory.

## Citation

If you use this code, please cite the original paper:
Medialdea L., Arribas-Gil A., Pérez-Romero A., Gómez A. (2024). "Classification of nutritional status of children from Geometric Morphometrics: an approach to out-of-sample data"
