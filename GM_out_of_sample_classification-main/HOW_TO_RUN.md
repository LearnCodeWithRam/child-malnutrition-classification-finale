# How to Run the Code

## 🚀 Fastest Way (Windows)

1. Open Command Prompt in this folder
2. Double-click `setup_and_run.bat` or run:
   ```cmd
   setup_and_run.bat
   ```

This will install dependencies and run the example with sample data.

---

## 📋 Step-by-Step Instructions

### Prerequisites

- Python 3.7 or higher installed
- R installed (only if you want to use real gorilla skull data)

### Step 1: Install Python Dependencies

Open Command Prompt (Windows) or Terminal (Mac/Linux) in this folder:

```bash
pip install -r requirements.txt
```

This installs: numpy, scipy, scikit-learn, pandas, matplotlib

### Step 2: Choose Your Data Source

#### Option A: Use Sample Data (Quick Test)

Just run:
```bash
python run_example.py
```

The code will automatically generate sample data for testing.

#### Option B: Use Real Gorilla Skull Data

1. **Export data from R:**
   
   Open R and run:
   ```r
   source("export_data_from_R.R")
   ```
   
   This creates `gorf_data.npy` and `gorm_data.npy` files.

2. **Run Python code:**
   ```bash
   python run_example.py
   ```
   
   The code will automatically detect and use the real data.

---

## 📁 Available Scripts

### 1. `run_example.py` - Quick Demo
```bash
python run_example.py
```
- Runs complete classification pipeline
- Shows confusion matrices and accuracy
- Creates visualization plots
- **Best for**: First-time users, testing installation

### 2. `code_out_of_sample_class.py` - Main Analysis
```bash
python code_out_of_sample_class.py
```
- Full analysis of gorilla skull data
- Multiple classification methods (LDA, LR, k-NN)
- Alignment to mean and median
- **Best for**: Reproducing paper results

### 3. `Simulation_study.py` - Full Simulation
```bash
python Simulation_study.py
```
- Runs 100 simulation iterations
- Tests 5 different scenarios
- Takes 10-30 minutes to complete
- **Best for**: Comprehensive performance evaluation

### 4. `Simulation_study_example_visualization.py` - Visual Simulation
```bash
python Simulation_study_example_visualization.py
```
- Single simulation run with plots
- Shows GPA and wGPA alignment results
- Compares in-sample vs out-of-sample
- **Best for**: Understanding the methods visually

---

## 🔧 Customization

### Change k-NN Neighbors

Edit the `knnk` parameter:
```python
class_results = loo_cv_oos_classification(n, k, sk, true_class, knnk=3)  # Use k=3
```

### Disable Allometric Effect Correction

Set `all_ef=False`:
```python
class_results = loo_cv_oos_classification(n, k, sk, true_class, all_ef=False)
```

### Change Alignment Method

Use weighted GPA:
```python
class_results = loo_cv_oos_classification(n, k, sk, true_class, method="wGPA")
```

### Test Different Scenarios

In `Simulation_study.py`, uncomment different scenario blocks:
```python
# Scenario 2: more distant group means
ms1 = Proc_F['mshape']
lndk7 = np.zeros((k, 2))
lndk7[6, 0] = 20
ms2 = Proc_M['mshape'] + lndk7
s1 = s2 = 5 * np.mean(np.abs(ms1 + ms2) / 2) / n1
```

---

## 📊 Expected Output

### Console Output
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
true_class  F   M
mean_registration
F          28   3
M           2  26
{'acc': 0.9153, 'sens': 0.9333, 'spec': 0.8966}

[... more results ...]
```

### Visual Output
- Scatter plots of aligned shapes
- Mean shape (black line)
- Median shape (red line)
- Individual configurations (gray points)

---

## ❓ Troubleshooting

### "pip is not recognized"

Install pip or use:
```bash
python -m pip install -r requirements.txt
```

### "No module named 'aux_functions'"

Make sure you're in the correct directory:
```bash
cd GM_out_of_sample_classification-main
python run_example.py
```

### "FileNotFoundError: gorf_data.npy"

This is normal! The code will use sample data automatically. To use real data, run `export_data_from_R.R` in R first.

### Plots don't show

If using Jupyter or remote environment:
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

### Slow performance

- Reduce `n_runs` in `Simulation_study.py` (e.g., from 100 to 10)
- Use smaller sample sizes
- Disable plotting during simulations

---

## 📚 File Structure

```
GM_out_of_sample_classification-main/
├── aux_functions.py              # Core functions
├── code_out_of_sample_class.py   # Main analysis
├── Simulation_study.py           # Full simulation
├── Simulation_study_example_visualization.py  # Visual simulation
├── run_example.py                # Quick demo script
├── export_data_from_R.R          # R data export script
├── setup_and_run.bat             # Windows quick start
├── requirements.txt              # Python dependencies
├── HOW_TO_RUN.md                 # This file
├── QUICKSTART.md                 # Quick reference
└── README_PYTHON.md              # Detailed documentation
```

---

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| Quick test | `python run_example.py` |
| Export R data | `Rscript export_data_from_R.R` |
| Full analysis | `python code_out_of_sample_class.py` |
| Run simulation | `python Simulation_study.py` |
| Visual demo | `python Simulation_study_example_visualization.py` |

---

## 💡 Tips

1. **Start with `run_example.py`** to verify everything works
2. **Use sample data first** before exporting real data
3. **Check console output** for progress and errors
4. **Close plot windows** to continue execution
5. **Read README_PYTHON.md** for detailed documentation

---

## 📞 Need Help?

1. Check `README_PYTHON.md` for detailed documentation
2. Review `QUICKSTART.md` for common issues
3. Verify Python version: `python --version` (need 3.7+)
4. Verify packages: `pip list | grep numpy`

---

## ✅ Success Checklist

- [ ] Python 3.7+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Can run `python run_example.py` without errors
- [ ] See classification results in console
- [ ] (Optional) Exported real data from R
- [ ] (Optional) Plots display correctly

If all checked, you're ready to use the code! 🎉
