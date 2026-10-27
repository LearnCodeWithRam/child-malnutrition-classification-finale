# Quick Start Guide

## Running the Python Code

### Option 1: Quick Test with Sample Data

1. **Install dependencies:**
   ```bash
   cd GM_out_of_sample_classification-main
   pip install -r requirements.txt
   ```

2. **Run the example:**
   ```bash
   python run_example.py
   ```

   This will run with synthetic sample data to demonstrate the code works.

### Option 2: Use Real Gorilla Skull Data

#### Step A: Export Data from R

First, run this in R to export the gorilla skull data:

```r
# Install required R packages if needed
install.packages("shapes")
install.packages("reticulate")

# Load libraries
library(shapes)
library(reticulate)

# Import numpy through reticulate
np <- import("numpy")

# Export the gorilla skull data
np$save("gorf_data.npy", gorf.dat)
np$save("gorm_data.npy", gorm.dat)
```

This creates two files: `gorf_data.npy` and `gorm_data.npy`

#### Step B: Move Data Files

Move the `.npy` files to the `GM_out_of_sample_classification-main` directory.

#### Step C: Run Python Code

```bash
cd GM_out_of_sample_classification-main
python run_example.py
```

The code will automatically detect and use the real data.

---

## Running Individual Scripts

### 1. Main Classification Example

```bash
python code_out_of_sample_class.py
```

Performs classification on gorilla skull data with various methods.

### 2. Simulation Study

```bash
python Simulation_study.py
```

Runs 100 simulation iterations (takes several minutes).

### 3. Simulation with Visualization

```bash
python Simulation_study_example_visualization.py
```

Runs one simulation with plots showing the alignment results.

---

## Alternative: Export Data as CSV from R

If you prefer CSV format:

```r
library(shapes)

# Function to convert 3D array to data frame
array_to_df <- function(arr) {
  k <- dim(arr)[1]
  n <- dim(arr)[3]
  df <- data.frame()
  for (i in 1:n) {
    temp <- data.frame(
      individual = i,
      landmark = 1:k,
      x = arr[, 1, i],
      y = arr[, 2, i]
    )
    df <- rbind(df, temp)
  }
  return(df)
}

# Export
write.csv(array_to_df(gorf.dat), "gorf_data.csv", row.names = FALSE)
write.csv(array_to_df(gorm.dat), "gorm_data.csv", row.names = FALSE)
```

Then load in Python:

```python
import pandas as pd
import numpy as np

def load_csv_data(filename, k=8):
    df = pd.read_csv(filename)
    n = df['individual'].max()
    data = np.zeros((k, 2, n))
    for i in range(1, n+1):
        subset = df[df['individual'] == i]
        data[:, 0, i-1] = subset['x'].values
        data[:, 1, i-1] = subset['y'].values
    return data

sk_F = load_csv_data("gorf_data.csv")
sk_M = load_csv_data("gorm_data.csv")
```

---

## Troubleshooting

### Import Errors

If you get import errors:
```bash
pip install --upgrade numpy scipy scikit-learn pandas matplotlib
```

### "No module named 'aux_functions'"

Make sure you're running from the `GM_out_of_sample_classification-main` directory:
```bash
cd GM_out_of_sample_classification-main
python run_example.py
```

### Data Not Found

The code will use sample data if real data isn't available. To use real data, follow "Option 2" above.

---

## Expected Output

When running `run_example.py`, you should see:

```
Loading data...
Using sample data for demonstration.

Data dimensions:
  Landmarks: 8
  Females: 30
  Males: 29

1. Running Procrustes alignment on female skulls...
2. Running Procrustes alignment on male skulls...
3. Running Procrustes alignment on combined sample...

4. Running leave-one-out cross-validation classification...

============================================================
CLASSIFICATION RESULTS
============================================================

LDA mean alignment
...
[Confusion matrices and accuracy metrics]
...

Classification complete!
```

---

## Next Steps

- Modify scenarios in `Simulation_study.py` (uncomment different scenarios)
- Adjust parameters: `knnk` (k-NN neighbors), `all_ef` (allometric effect)
- Use your own landmark data (same format: k×2×n arrays)
