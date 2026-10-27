"""
Code for "Classification of nutritional status of children from 
Geometric Morphometrics: an approach to out-of-sample data" (2024)
Medialdea L., Arribas-Gil A., Pérez-Romero A., Gómez A. 

Full simulation study
Requires libraries: numpy, scipy, scikit-learn, pandas
"""

import numpy as np
from aux_functions import (
    proc_gpa, loo_cv_oos_classification,
    append_tables_oos, acc2_list
)

# Note: This requires gorilla skull data from R 'shapes' package
# Export from R: sk_F = shapes::gorf.dat, sk_M = shapes::gorm.dat

def load_gorilla_data():
    """Load gorilla skull data - replace with actual data loading"""
    print("Note: You need to provide gorilla skull data")
    return None, None

sk_F, sk_M = load_gorilla_data()

if sk_F is not None and sk_M is not None:
    n_runs = 100  # number of simulation runs for each setting
    joinline = np.array([0, 5, 6, 7, 1, 2, 3, 4, 0])
    jl = joinline[:-1]
    
    ## 1. Loading female and male gorilla skull data + simulation settings
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

    ###################################################################
    # # Scenario 2: more distant group means, same dispersion 
    # ms1 = Proc_F['mshape']
    # lndk7 = np.zeros((k, 2))
    # lndk7[6, 0] = 20
    # ms2 = Proc_M['mshape'] + lndk7
    # s1 = s2 = 5 * np.mean(np.abs(ms1 + ms2) / 2) / n1
    ###################################################################
    
    ###################################################################
    # # Scenario 3: same mean - different dispersion (ratio = 5) 
    # ms1 = ms2 = Proc_F['mshape']
    # s1 = 5 * np.mean(np.abs(ms1)) / n1
    # s2 = 5 * s1
    ###################################################################
    
    ###################################################################
    # # Scenario 4: same mean - different dispersion (ratio = 3) 
    # ms1 = ms2 = Proc_F['mshape']
    # s1 = 5 * np.mean(np.abs(ms1)) / n1
    # s2 = 3 * s1
    ###################################################################
    
    ###################################################################
    # # Scenario 5: same mean - different dispersion (ratio = 1.5) 
    # ms1 = ms2 = Proc_F['mshape']
    # s1 = 5 * np.mean(np.abs(ms1)) / n1
    # s2 = 1.5 * s1
    ###################################################################
    
    tables_GPA = [np.zeros((2, 2)) for _ in range(6)]
    tables_wGPA = [np.zeros((2, 2)) for _ in range(6)]
    tables_SRVF = [np.zeros((2, 2)) for _ in range(6)]
    
    for j in range(n_runs):
        print(f"Simulation run {j+1}/{n_runs}")
        
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
        
        ## 3. Out-of-sample classification with LOO CV
        true_class = np.array(['G1'] * n1 + ['G2'] * n2)
        
        class_results_GPA = loo_cv_oos_classification(n, k, G, true_class, 
                                                       all_ef=True, knnk=5)
        class_results_wGPA = loo_cv_oos_classification(n, k, G, true_class, 
                                                        method="wGPA", all_ef=True, knnk=5)
        
        # Note: SRVF alignment requires fdasrvf package - not implemented here
        # class_results_SRVF = srvf_loo_cv_oos_classification(n, k, G[jl, :, :], 
        #                                                      true_class, all_ef=True, knnk=5)
        
        ## 4. Classification results
        tables_GPA = append_tables_oos(class_results_GPA, tables_GPA)
        tables_wGPA = append_tables_oos(class_results_wGPA, tables_wGPA)
        # tables_SRVF = append_tables_oos(class_results_SRVF, tables_SRVF)
    
    res_GPA = acc2_list(tables_GPA, n_runs)
    res_wGPA = acc2_list(tables_wGPA, n_runs)
    # res_SRVF = acc2_list(tables_SRVF, n_runs)
    
    print("\n=== GPA Results ===")
    for i, res in enumerate(res_GPA):
        print(f"Method {i+1}: {res}")
    
    print("\n=== wGPA Results ===")
    for i, res in enumerate(res_wGPA):
        print(f"Method {i+1}: {res}")
