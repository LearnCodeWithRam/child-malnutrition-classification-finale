"""
Code for "Classification of nutritional status of children from 
Geometric Morphometrics: an approach to out-of-sample data" (2024)
Medialdea L., Arribas-Gil A., Pérez-Romero A., Gómez A. 

Simulation study - visualization of the different scenarios
Requires libraries: numpy, scipy, scikit-learn, pandas, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from aux_functions import (
    proc_gpa, proc_wgpa_silent, rotate_skulls, plot_shapes,
    loo_cv_is_classification, loo_cv_oos_classification,
    summary_results_is, summary_results_oos
)

# Note: This requires gorilla skull data from R 'shapes' package

def load_gorilla_data():
    """Load gorilla skull data - replace with actual data loading"""
    print("Note: You need to provide gorilla skull data")
    return None, None

sk_F, sk_M = load_gorilla_data()

if sk_F is not None and sk_M is not None:
    ## 1. Loading female and male gorilla skull data
    k = sk_F.shape[0]  # number of landmarks = 8
    
    Proc_F = proc_gpa(sk_F)
    Proc_M = proc_gpa(sk_M)
    
    n1 = n2 = 50
    
    ## 2. Data generation
    # 2.a Parameter settings for 5 different scenarios
    
    ###################################################################
    # Scenario 1: different mean, same dispersion
    ms1 = Proc_F['mshape']
    ms2 = Proc_M['mshape']
    s1 = s2 = 5 * np.mean(np.abs(ms1 + ms2) / 2) / n1
    ###################################################################
    
    # Other scenarios (commented out - uncomment to use)
    # See Simulation_study.py for other scenario definitions
    
    # 2.b Generate aligned data
    G1 = np.zeros((k, 2, n1))
    G2 = np.zeros((k, 2, n2))
    
    for i in range(n1):
        G1[:, :, i] = ms1 + np.random.randn(k, 2) * s1
        G2[:, :, i] = ms2 + np.random.randn(k, 2) * s2

    # 2.c Add rotation, translation and scaling
    G = np.concatenate([G1, G2], axis=2)
    n = G.shape[2]
    
    for i in range(n):
        # Rotation matrix
        theta = 2 * np.pi * np.random.rand()
        Rot = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
        
        # Scaling factor
        alpha = np.random.uniform(0, 10)
        
        # Translation matrix
        beta = np.tile(np.random.uniform(-5, 5, 2), (k, 1))
        
        G[:, :, i] = beta + alpha * (G[:, :, i] @ Rot)
    
    # 2.d Full Alignment (GPA, wGPA) of the whole sample
    joinline = np.array([0, 5, 6, 7, 1, 2, 3, 4, 0])
    
    # 2.d.1 GPA
    Proc = proc_gpa(G)
    Proc_rotated = rotate_skulls(Proc['rotated'], Proc['mshape'])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plt.sca(axes[0])
    plot_shapes(Proc_rotated[:, :, :n1], k, n1, joinline=joinline,
                maintext="Group 1 after GPA", lines=False, med_pw=False, 
                median=True, mean=True)
    
    plt.sca(axes[1])
    plot_shapes(Proc_rotated[:, :, n1:], k, n2, joinline=joinline,
                maintext="Group 2 after GPA", lines=False, med_pw=False, 
                median=True, mean=True)
    plt.tight_layout()
    plt.show()
    
    plot_shapes(Proc_rotated, k, n, joinline=joinline,
                maintext="Both groups (G1 + G2) after GPA", 
                med_pw=True, median=True)
    
    # 2.d.2 wGPA
    WProc = proc_wgpa_silent(G)
    WProc_rotated = rotate_skulls(WProc['rotated'], WProc['mshape'])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plt.sca(axes[0])
    plot_shapes(WProc_rotated[:, :, :n1], k, n1, joinline=joinline,
                maintext="Group 1 after wGPA", lines=False, med_pw=False, 
                median=True, mean=True)
    
    plt.sca(axes[1])
    plot_shapes(WProc_rotated[:, :, n1:], k, n2, joinline=joinline,
                maintext="Group 2 after wGPA", lines=False, med_pw=False, 
                median=True, mean=True)
    plt.tight_layout()
    plt.show()
    
    plot_shapes(WProc_rotated, k, n, joinline=joinline,
                maintext="Both groups (G1 + G2) after wGPA", 
                med_pw=True, median=True)
    
    # Note: SRVF alignment requires fdasrvf package - not implemented here
    
    ## 3. Classification with GPA alignment
    true_class = np.array(['G1'] * n1 + ['G2'] * n2)
    
    # In-sample classification
    print("\n=== In-sample classification (GPA) ===")
    class_results_IS = loo_cv_is_classification(n, k, G, true_class, 
                                                 all_ef=True, knnk=5)
    summary_results_is(class_results_IS)
    
    # Out-of-sample classification
    print("\n=== Out-of-sample classification (GPA) ===")
    class_results_OOS = loo_cv_oos_classification(n, k, G, true_class, 
                                                   all_ef=True, knnk=5)
    summary_results_oos(class_results_OOS)
    
    ## 4. Classification with wGPA alignment
    
    # In-sample classification
    print("\n=== In-sample classification (wGPA) ===")
    class_results_wGPA_IS = loo_cv_is_classification(n, k, G, true_class, 
                                                      method="wGPA", all_ef=True, knnk=5)
    summary_results_is(class_results_wGPA_IS)
    
    # Out-of-sample classification
    print("\n=== Out-of-sample classification (wGPA) ===")
    class_results_wGPA_OOS = loo_cv_oos_classification(n, k, G, true_class, 
                                                        method="wGPA", all_ef=True, knnk=5)
    summary_results_oos(class_results_wGPA_OOS)
    
    # Note: SRVF classification requires fdasrvf package - not implemented here
