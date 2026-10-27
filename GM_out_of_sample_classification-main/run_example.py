"""
Example script to run the classification code with sample data
"""

import numpy as np
from aux_functions import (
    proc_gpa, rotate_skulls, plot_shapes, 
    loo_cv_oos_classification, summary_results_oos
)

# Create sample gorilla skull data (replace with real data)
def create_sample_data():
    """
    Create sample data for demonstration
    In practice, export real data from R using:
    
    library(shapes)
    library(reticulate)
    np <- import("numpy")
    np$save("gorf_data.npy", gorf.dat)
    np$save("gorm_data.npy", gorm.dat)
    """
    np.random.seed(42)
    
    # Female skulls: 8 landmarks, 2D coordinates, 30 individuals
    k, n1 = 8, 30
    sk_F = np.random.randn(k, 2, n1) * 0.1
    # Add some structure
    for i in range(n1):
        sk_F[:, :, i] += np.array([[0, 0], [1, 0], [2, 0], [2, 1], 
                                     [2, 2], [1, 2], [0, 2], [0, 1]])
    
    # Male skulls: 8 landmarks, 2D coordinates, 29 individuals
    n2 = 29
    sk_M = np.random.randn(k, 2, n2) * 0.1
    # Add some structure (slightly different from females)
    for i in range(n2):
        sk_M[:, :, i] += np.array([[0, 0], [1.2, 0], [2.2, 0], [2.2, 1], 
                                     [2.2, 2.2], [1.2, 2.2], [0, 2.2], [0, 1]])
    
    return sk_F, sk_M

# Load or create data
print("Loading data...")
try:
    # Try to load real data if available
    sk_F = np.load("gorf_data.npy")
    sk_M = np.load("gorm_data.npy")
    print("Loaded real gorilla skull data")
except FileNotFoundError:
    print("Real data not found. Using sample data for demonstration.")
    print("To use real data, export from R (see README_PYTHON.md)")
    sk_F, sk_M = create_sample_data()

k = sk_F.shape[0]  # number of landmarks
n1 = sk_F.shape[2]  # number of females
n2 = sk_M.shape[2]  # number of males

print(f"\nData dimensions:")
print(f"  Landmarks: {k}")
print(f"  Females: {n1}")
print(f"  Males: {n2}")

## Full GPA on each group
print("\n1. Running Procrustes alignment on female skulls...")
Proc_F = proc_gpa(sk_F)
Proc_F_rotated = rotate_skulls(Proc_F['rotated'], Proc_F['mshape'])

print("2. Running Procrustes alignment on male skulls...")
Proc_M = proc_gpa(sk_M)
Proc_M_rotated = rotate_skulls(Proc_M['rotated'], Proc_M['mshape'])

## Full GPA of combined sample
print("3. Running Procrustes alignment on combined sample...")
sk = np.concatenate([sk_F, sk_M], axis=2)
Proc = proc_gpa(sk)
n = sk.shape[2]
Proc_rotated = rotate_skulls(Proc['rotated'], Proc['mshape'])

## Classification with leave-one-out cross validation
print("\n4. Running leave-one-out cross-validation classification...")
true_class = np.array(['F'] * n1 + ['M'] * n2)

class_results = loo_cv_oos_classification(n, k, sk, true_class, all_ef=True, knnk=5)

## Display results
print("\n" + "="*60)
print("CLASSIFICATION RESULTS")
print("="*60)

summary_results_oos(class_results)

print("\n" + "="*60)
print("Classification complete!")
print("="*60)

# Optional: Visualize results
try:
    import matplotlib.pyplot as plt
    
    print("\nGenerating visualizations...")
    joinline = np.array([0, 5, 6, 7, 1, 2, 3, 4, 0])
    
    # Plot combined sample
    plot_shapes(Proc_rotated, k, n, joinline=joinline,
                maintext="Gorilla skulls (F+M) after GPA", 
                med_pw=True, median=True)
    
    print("Visualization complete. Close the plot window to exit.")
    
except Exception as e:
    print(f"\nVisualization skipped: {e}")

print("\nDone!")
