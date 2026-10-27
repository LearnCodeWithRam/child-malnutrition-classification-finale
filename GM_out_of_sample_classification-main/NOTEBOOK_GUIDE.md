# 📓 Jupyter Notebook Guide

## ✅ Notebook Successfully Created!

The file `GM_OUT_Geometric_Classification.ipynb` is now ready to use.

## 🚀 How to Run

### Option 1: Using Jupyter Notebook

```bash
# Install Jupyter if needed
pip install jupyter nbformat

# Navigate to folder
cd GM_out_of_sample_classification-main

# Launch Jupyter
jupyter notebook

# Click on GM_OUT_Geometric_Classification.ipynb
```

### Option 2: Using Jupyter Lab

```bash
# Install Jupyter Lab
pip install jupyterlab

# Launch
jupyter lab

# Open the notebook from the file browser
```

### Option 3: Using VS Code

1. Open VS Code
2. Install "Jupyter" extension
3. Open the `.ipynb` file
4. Click "Run All" or run cells individually

## 📚 What's Inside (17 Cells)

### Part 1: Setup & Functions (Cells 1-3)
- Import libraries
- Define ALL helper functions (Procrustes, GPA, classification)
- Load real gorilla data (or sample data as fallback)

### Part 2: Data Visualization (Cells 4-5)
- Visualize raw data before alignment
- Apply GPA to each group separately
- Show aligned results

### Part 3: Combined Analysis (Cells 6-8)
- Apply GPA to combined sample (Female + Male)
- Visualize group differences
- Compare mean shapes

### Part 4: Classification (Cells 9-12)
- Out-of-sample LOOCV classification
- LDA, Logistic Regression, k-NN results
- Confusion matrices
- Accuracy comparison

### Part 5: Summary (Cell 13)
- Key takeaways
- What was removed vs what remains
- How to use with arm data

## 🎯 Key Features

✅ **Self-contained** - All functions defined in notebook  
✅ **Real data support** - Uses actual gorilla skull data if available  
✅ **Automatic fallback** - Creates sample data if real data not found  
✅ **Visual** - Plots at every major step  
✅ **Educational** - Understand what's happening under the hood  
✅ **Flexible** - Same code works for gorilla skulls OR arm contours

## 📊 Using Real Gorilla Data

The notebook automatically tries to load real gorilla skull data:

### Option 1: Export from R (Recommended)

```r
# In R, run:
source("export_data_from_R.R")
```

This creates:
- `gorf_data.npy` (female gorilla skulls)
- `gorm_data.npy` (male gorilla skulls)

### Option 2: Manual Export

```r
library(shapes)
library(reticulate)

np <- import("numpy")
np$save("gorf_data.npy", gorf.dat)
np$save("gorm_data.npy", gorm.dat)
```

### What Happens:

- ✅ **If files exist**: Loads real gorilla data
- ⚠️ **If files missing**: Uses sample data (still works!)

## 🔄 Switching to Arm Data (Future)

When you receive arm contour data:

1. **Export arm landmarks** as `.npy` files:
   ```python
   np.save("healthy_arms.npy", healthy_arm_data)
   np.save("sam_arms.npy", sam_arm_data)
   ```

2. **Update data loading cell**:
   ```python
   sk_F = np.load("healthy_arms.npy")  # Instead of gorf_data.npy
   sk_M = np.load("sam_arms.npy")      # Instead of gorm_data.npy
   ```

3. **Update labels**:
   ```python
   true_class = np.array(['Healthy'] * n1 + ['SAM'] * n2)
   ```

4. **Run notebook** - Everything else stays the same!

The geometric morphometrics pipeline is universal! 🎉

## 📊 Expected Output

When you run all cells, you'll see:
- Original arm shapes (healthy vs SAM)
- Transformation effects (the problem)
- Step-by-step Procrustes analysis
- Before/after comparisons
- GPA convergence
- Classification results with accuracies

## 🔧 Troubleshooting

### "No module named 'nbformat'"
```bash
pip install nbformat
```

### "Kernel not found"
```bash
python -m ipykernel install --user
```

### Plots not showing
Add this at the top of a cell:
```python
%matplotlib inline
```

### Want to recreate the notebook?
```bash
python create_notebook.py
```

## 📝 Customization

You can easily modify:
- Number of samples (`n_healthy`, `n_sam`)
- Landmark positions (change `healthy_arm`, `sam_arm`)
- Classification methods (add more classifiers)
- Visualization styles (matplotlib parameters)

## 🎓 Learning Path

1. **First run:** Execute all cells to see the complete pipeline
2. **Second run:** Read explanations and understand the math
3. **Third run:** Modify parameters and see effects
4. **Fourth run:** Try with your own data!

## 📚 Additional Resources

- `aux_functions.py` - Standalone Python functions
- `run_example.py` - Command-line version
- `README_PYTHON.md` - Detailed documentation
- `HOW_TO_RUN.md` - Complete setup guide

## ✨ What Makes This Special

Unlike typical notebooks that import custom modules, this notebook:
- ✅ Has ALL code embedded (no external imports)
- ✅ Can be shared as a single file
- ✅ Works immediately after opening
- ✅ Perfect for learning and teaching

## 🎉 You're Ready!

Open the notebook and start exploring geometric morphometrics!

```bash
jupyter notebook GM_OUT_Geometric_Classification.ipynb
```

Happy analyzing! 🔬📊🎯
