"""
Code for "Classification of nutritional status of children from 
Geometric Morphometrics: an approach to out-of-sample data" (2024)
Medialdea L., Arribas-Gil A., Pérez-Romero A., Gómez A. 

Main code and example using data for the analysis of sexual dimorphism in gorillas skulls
Requires libraries: numpy, scipy, scikit-learn, pandas, matplotlib
"""

import numpy as np
import pandas as pd
from aux_functions import (
    proc_gpa, rotate_skulls, plot_shapes, 
    loo_cv_oos_classification
)

# Note: This example requires gorilla skull data from the R 'shapes' package
# You'll need to export the data from R or use equivalent data
# Example data structure: sk_F and sk_M should be 3D arrays of shape (k, 2, n)
# where k=8 landmarks, 2 dimensions (x,y), n=number of individuals

def load_gorilla_data():
    """
    Load gorilla skull data
    In R, this would be: sk_F = shapes::gorf.dat, sk_M = shapes::gorm.dat
    You need to export this data from R or provide equivalent data
    """
    # Placeholder - replace with actual data loading
    print("Note: You need to provide gorilla skull data")
    print("Export from R using: write.csv() or save as .npy files")
    return None, None

# Load data
sk_F, sk_M = load_gorilla_data()

if sk_F is not None and sk_M is not None:
    k = sk_F.shape[0]  # number of landmarks = 8
    n1 = sk_F.shape[2]  # number of females = 30
    n2 = sk_M.shape[2]  # number of males = 29
    
    ## 2. Full GPA
    # 2.a Full GPA on each group separately
    
    # Female skulls Procrustes coordinates
    Proc_F = proc_gpa(sk_F)
    Proc_F_rotated = rotate_skulls(Proc_F['rotated'], Proc_F['mshape'])
    
    # Visualization
    plot_shapes(Proc_F_rotated, k, n1, joinline=np.array([0, 5, 6, 7, 1, 2, 3, 4, 0]),
                maintext="Female Gorillas skulls", med_pw=True, median=True)
    
    # Male skulls Procrustes coordinates
    Proc_M = proc_gpa(sk_M)
    Proc_M_rotated = rotate_skulls(Proc_M['rotated'], Proc_M['mshape'])
    
    # Visualization
    plot_shapes(Proc_M_rotated, k, n2, joinline=np.array([0, 5, 6, 7, 1, 2, 3, 4, 0]),
                maintext="Male Gorillas skulls", med_pw=True, median=True)

    # 2.b Full GPA of the whole sample
    sk = np.concatenate([sk_F, sk_M], axis=2)
    Proc = proc_gpa(sk)
    n = sk.shape[2]
    Proc_rotated = rotate_skulls(Proc['rotated'], Proc['mshape'])
    
    plot_shapes(Proc_rotated, k, n, joinline=np.array([0, 5, 6, 7, 1, 2, 3, 4, 0]),
                maintext="Gorilla skulls (F+M)", med_pw=True, median=True)
    
    ## 3. Classification with leave-one-out cross validation
    true_class = np.array(['F'] * n1 + ['M'] * n2)
    
    class_results = loo_cv_oos_classification(n, k, sk, true_class, all_ef=True, knnk=5)
    
    ## 4. Classification results
    
    # 4.1 LDA with fOPA to sample mean
    print("LDA with fOPA to sample mean:")
    print(pd.crosstab(class_results['predicted_class_LDA']['mean_registration'],
                      class_results['predicted_class_LDA']['true_class']))
    
    # 4.2 LDA with fOPA to sample median
    print("\nLDA with fOPA to sample median:")
    print(pd.crosstab(class_results['predicted_class_LDA']['median_registration'],
                      class_results['predicted_class_LDA']['true_class']))
    
    # 4.3 LR with fOPA to sample mean
    print("\nLR with fOPA to sample mean:")
    print(pd.crosstab(class_results['predicted_class_LR']['mean_registration'],
                      class_results['predicted_class_LR']['true_class']))
    
    # 4.4 LR with fOPA to sample median
    print("\nLR with fOPA to sample median:")
    print(pd.crosstab(class_results['predicted_class_LR']['median_registration'],
                      class_results['predicted_class_LR']['true_class']))
    
    # 4.5 k-NN with fOPA to sample mean
    print("\nk-NN with fOPA to sample mean:")
    print(pd.crosstab(class_results['predicted_class_knn']['mean_registration'],
                      class_results['predicted_class_knn']['true_class']))
    
    # 4.6 k-NN with fOPA to sample median
    print("\nk-NN with fOPA to sample median:")
    print(pd.crosstab(class_results['predicted_class_knn']['median_registration'],
                      class_results['predicted_class_knn']['true_class']))
