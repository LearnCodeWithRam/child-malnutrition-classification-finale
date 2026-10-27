"""
Code for "Classification of nutritional status of children from 
Geometric Morphometrics: an approach to out-of-sample data" (2024)
Medialdea L., Arribas-Gil A., Pérez-Romero A., Gómez A. 

Auxiliary functions
Requires libraries: numpy, scipy, scikit-learn, pandas
"""

import numpy as np
import pandas as pd
from scipy.linalg import svd, eig
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')


def loo_cv_oos_classification(n, k, M, true_class, method="GPA", all_ef=True, knnk=5):
    """Out-of-sample LOOCV classification with GPA or wGPA alignment"""
    
    predicted_class_LDA = pd.DataFrame({
        'mean_registration': [''] * n,
        'median_registration': [''] * n,
        'true_class': true_class
    })
    predicted_class_LR = pd.DataFrame({
        'mean_registration': [''] * n,
        'median_registration': [''] * n,
        'true_class': true_class
    })
    predicted_class_knn = pd.DataFrame({
        'mean_registration': [''] * n,
        'median_registration': [''] * n,
        'true_class': true_class
    })
    
    g1 = np.unique(true_class)[0]
    g2 = np.unique(true_class)[1]
    
    size = np.zeros(n)
    for i in range(n):
        mm = np.tile(np.mean(M[:, :, i], axis=0), (k, 1))
        size[i] = np.sqrt(np.sum((M[:, :, i] - mm) ** 2))
    
    for i in range(n):
        # Procrustes configuration removing individual i
        M_ = np.delete(M, i, axis=2)
        
        if method == "wGPA":
            Proc_ = proc_wgpa_silent(M_)
            Sigmak = Proc_['Sigmak']
            eig_vals = np.linalg.eigvals(Sigmak)
            while np.min(eig_vals) == 0:
                Sigmak = Sigmak + 1e-5 * np.diag(np.ones(k) * np.random.randn())
                eig_vals = np.linalg.eigvals(Sigmak)
        else:
            Proc_ = proc_gpa(M_)
        
        ms = Proc_['mshape']
        median_shape = compute_functional_median(Proc_['rotated'], k)
        
        if all_ef:
            X_proc_, reg_allom = compute_allometric_residuals(Proc_['rotated'], size[np.arange(n) != i], n - 1, k)
        else:
            X_proc_ = np.array([Proc_['rotated'][:, :, j].T.flatten() for j in range(n - 1)])
        
        # Training
        X_class = pd.DataFrame(X_proc_)
        X_class['cl'] = true_class[np.arange(n) != i]
        
        mod = LinearDiscriminantAnalysis()
        mod.fit(X_proc_, true_class[np.arange(n) != i])
        
        X_class_lr = X_class.copy()
        X_class_lr['cl'] = (X_class_lr['cl'] == g2).astype(int)
        lr = LogisticRegression(max_iter=1000, solver='lbfgs')
        lr.fit(X_proc_, X_class_lr['cl'])
        
        # Alignment of new individual to sample mean
        if method == "wGPA":
            testProc_new_mean = wopa(ms, M[:, :, i], Sigmak).T.flatten().reshape(1, -1)
        else:
            testProc_new_mean = proc_opa(ms, M[:, :, i]).T.flatten().reshape(1, -1)
        
        # Alignment of new individual to functional median
        if method == "wGPA":
            testProc_new_median = wopa(median_shape, M[:, :, i], Sigmak).T.flatten().reshape(1, -1)
        else:
            testProc_new_median = proc_opa(median_shape, M[:, :, i]).T.flatten().reshape(1, -1)
        
        if all_ef:
            testProc_new_mean = testProc_new_mean - reg_allom['intercept'] - reg_allom['coef'] * np.log(size[i])
            testProc_new_median = testProc_new_median - reg_allom['intercept'] - reg_allom['coef'] * np.log(size[i])
        
        # Classification with alignment to mean
        P_LDAmean = mod.predict(testProc_new_mean)[0]
        lr_mean = lr.predict(testProc_new_mean)[0]
        prob_lr_mean = g1 if lr_mean == 0 else g2
        
        knn = KNeighborsClassifier(n_neighbors=knnk)
        knn.fit(X_proc_, true_class[np.arange(n) != i])
        pred_class_meanknn = knn.predict(testProc_new_mean)[0]
        
        # Classification with alignment to median
        P_LDAmedian = mod.predict(testProc_new_median)[0]
        lr_median = lr.predict(testProc_new_median)[0]
        prob_lr_median = g1 if lr_median == 0 else g2
        
        pred_class_medianknn = knn.predict(testProc_new_median)[0]
        
        predicted_class_LDA.loc[i, ['mean_registration', 'median_registration']] = [P_LDAmean, P_LDAmedian]
        predicted_class_LR.loc[i, ['mean_registration', 'median_registration']] = [prob_lr_mean, prob_lr_median]
        predicted_class_knn.loc[i, ['mean_registration', 'median_registration']] = [pred_class_meanknn, pred_class_medianknn]
    
    return {
        'predicted_class_LDA': predicted_class_LDA,
        'predicted_class_LR': predicted_class_LR,
        'predicted_class_knn': predicted_class_knn
    }


def proc_gpa(X, scale=True):
    """Generalized Procrustes Analysis"""
    k, m, n = X.shape
    
    # Center configurations
    X_centered = np.zeros_like(X)
    for i in range(n):
        X_centered[:, :, i] = X[:, :, i] - np.mean(X[:, :, i], axis=0)
    
    # Scale to unit centroid size
    if scale:
        for i in range(n):
            cs = np.sqrt(np.sum(X_centered[:, :, i] ** 2))
            X_centered[:, :, i] = X_centered[:, :, i] / cs
    
    # Iterative alignment
    mshape = np.mean(X_centered, axis=2)
    for iteration in range(10):
        mshape_old = mshape.copy()
        for i in range(n):
            X_centered[:, :, i] = proc_opa(mshape, X_centered[:, :, i])
        mshape = np.mean(X_centered, axis=2)
        if np.sum((mshape - mshape_old) ** 2) < 1e-6:
            break
    
    return {'rotated': X_centered, 'mshape': mshape}


def proc_opa(A, B):
    """Ordinary Procrustes Analysis - align B to A"""
    # Center
    A_centered = A - np.mean(A, axis=0)
    B_centered = B - np.mean(B, axis=0)
    
    # SVD for rotation
    H = B_centered.T @ A_centered
    U, _, Vt = svd(H)
    R = Vt.T @ U.T
    
    # Apply rotation
    B_aligned = B_centered @ R
    
    return B_aligned


def proc_wgpa_silent(X, maxiterations=10, scale=True):
    """Weighted Generalized Procrustes Analysis (silent version)"""
    k, m, n = X.shape
    
    # Initialize with identity covariance
    Sigmak = np.eye(k)
    
    # Initial GPA
    Proc = proc_gpa(X, scale=scale)
    mu = Proc['mshape']
    
    # Iterative refinement
    for it in range(maxiterations):
        # Estimate covariance
        residuals = np.zeros((k, m, n))
        for i in range(n):
            residuals[:, :, i] = Proc['rotated'][:, :, i] - mu
        
        Sigmak_new = np.zeros((k, k))
        for d in range(m):
            Sigmak_new += np.cov(residuals[:, d, :])
        Sigmak_new = Sigmak_new / m
        
        # Make diagonal
        Sigmak_new = np.diag(np.diag(Sigmak_new))
        
        if np.linalg.norm(Sigmak - Sigmak_new) < 1e-5:
            break
        
        Sigmak = Sigmak_new
    
    return {'rotated': Proc['rotated'], 'mshape': mu, 'Sigmak': Sigmak}


def wopa(ms, x, Sigma):
    """Pairwise weighted Procrustes alignment"""
    k = x.shape[0]
    M1_ms = np.zeros((k, 2, 2))
    M1_ms[:, :, 0] = ms
    M1_ms[:, :, 1] = x
    
    wOPA = proc_wgpa_silent(M1_ms, maxiterations=5)
    x_reg = wOPA['rotated'][:, :, 1]
    ms_reg = wOPA['rotated'][:, :, 0]
    
    # Align back to original ms
    OPAms_result = proc_opa(ms, ms_reg)
    
    # Calculate scaling
    s = np.sqrt(np.sum(ms ** 2) / np.sum(ms_reg ** 2))
    
    # Get rotation from OPA
    H = ms_reg.T @ ms
    U, _, Vt = svd(H)
    R = Vt.T @ U.T
    
    ans = s * (x_reg - np.mean(ms_reg, axis=0)) @ R + np.mean(ms, axis=0)
    
    return ans


def compute_functional_median(rotated, k):
    """Compute functional median of shapes"""
    n = rotated.shape[2]
    
    # Simple implementation: use coordinate-wise median
    median_shape = np.median(rotated, axis=2)
    
    return median_shape


def compute_allometric_residuals(rotated, size, n, k):
    """Compute residuals after removing allometric effect"""
    X_proc = np.array([rotated[:, :, i].T.flatten() for i in range(n)])
    
    residuals = np.zeros_like(X_proc)
    intercepts = np.zeros(2 * k)
    coefs = np.zeros(2 * k)
    
    log_size = np.log(size)
    
    for j in range(2 * k):
        # Linear regression
        X_design = np.column_stack([np.ones(n), log_size])
        beta = np.linalg.lstsq(X_design, X_proc[:, j], rcond=None)[0]
        intercepts[j] = beta[0]
        coefs[j] = beta[1]
        residuals[:, j] = X_proc[:, j] - X_design @ beta
    
    return residuals, {'intercept': intercepts, 'coef': coefs}


def loo_cv_is_classification(n, k, M, true_class, method="GPA", all_ef=True, knnk=5):
    """In-sample LOOCV classification with (w)GPA alignment"""
    
    predicted_class_LDA = pd.DataFrame({
        'pred_class': [''] * n,
        'true_class': true_class
    })
    predicted_class_LR = pd.DataFrame({
        'pred_class': [''] * n,
        'true_class': true_class
    })
    predicted_class_knn = pd.DataFrame({
        'pred_class': [''] * n,
        'true_class': true_class
    })
    
    g1 = np.unique(true_class)[0]
    g2 = np.unique(true_class)[1]
    
    # Get global alignment
    if method == "wGPA":
        Proc = proc_wgpa_silent(M)
    else:
        Proc = proc_gpa(M)
    
    if all_ef:
        size = np.zeros(n)
        for i in range(n):
            mm = np.tile(np.mean(M[:, :, i], axis=0), (k, 1))
            size[i] = np.sqrt(np.sum((M[:, :, i] - mm) ** 2))
        X_proc, _ = compute_allometric_residuals(Proc['rotated'], size, n, k)
    else:
        X_proc = np.array([Proc['rotated'][:, :, i].T.flatten() for i in range(n)])
    
    X_class = pd.DataFrame(X_proc)
    X_class['cl'] = true_class
    
    X_classLR = X_class.copy()
    X_classLR['cl'] = (X_classLR['cl'] == g2).astype(int)
    
    for i in range(n):
        # Training
        train_idx = np.arange(n) != i
        
        mod = LinearDiscriminantAnalysis()
        mod.fit(X_proc[train_idx], true_class[train_idx])
        
        lr = LogisticRegression(max_iter=1000, solver='lbfgs')
        lr.fit(X_proc[train_idx], X_classLR.loc[train_idx, 'cl'])
        
        knn = KNeighborsClassifier(n_neighbors=knnk)
        knn.fit(X_proc[train_idx], true_class[train_idx])
        
        # Prediction
        P_LDA = mod.predict(X_proc[i].reshape(1, -1))[0]
        lr_p = lr.predict(X_proc[i].reshape(1, -1))[0]
        prob_lr = g1 if lr_p == 0 else g2
        pred_class_knn = knn.predict(X_proc[i].reshape(1, -1))[0]
        
        predicted_class_LDA.loc[i, 'pred_class'] = P_LDA
        predicted_class_LR.loc[i, 'pred_class'] = prob_lr
        predicted_class_knn.loc[i, 'pred_class'] = pred_class_knn
    
    return {
        'predicted_class_LDA': predicted_class_LDA,
        'predicted_class_LR': predicted_class_LR,
        'predicted_class_knn': predicted_class_knn
    }


def plot_shapes(M, k, n, joinline, maintext, lines=True, xxlim=None, yylim=None, 
                labs=None, color=None, mean=True, med_pw=True, median=True, ld=None):
    """Plot shapes with mean and median"""
    import matplotlib.pyplot as plt
    
    if n == 1:
        MM = np.zeros((k, 2, 1))
        MM[:, :, 0] = M
        M = MM
    
    if xxlim is None:
        xxlim = [np.min(M[:, 0, :]), np.max(M[:, 0, :])]
    if yylim is None:
        yylim = [np.min(M[:, 1, :]), np.max(M[:, 1, :])]
    if color is None:
        color = ['lightgray'] * n
    
    plt.figure(figsize=(8, 8))
    plt.scatter(M[:, 0, :].flatten(), M[:, 1, :].flatten(), 
                c='gray', s=10, alpha=0.5)
    
    if lines:
        for i in range(n):
            plt.plot(M[joinline, 0, i], M[joinline, 1, i], 
                    color=color[i], alpha=0.3)
    
    if mean:
        ms = np.mean(M, axis=2)
        plt.plot(ms[joinline, 0], ms[joinline, 1], 
                'k-', linewidth=2, label='Mean')
    
    if median:
        median_shape = np.median(M, axis=2)
        plt.plot(median_shape[joinline, 0], median_shape[joinline, 1], 
                'r-', linewidth=2, label='Median')
    
    plt.xlim(xxlim)
    plt.ylim(yylim)
    plt.title(maintext)
    plt.axis('equal')
    plt.legend()
    plt.show()


def rotate_skulls(Proc_rot, ms):
    """Rotate gorilla skulls so that line between landmarks 1 and 3 is parallel to horizontal axis"""
    n = Proc_rot.shape[2]
    alpha = ms[2, :] - ms[0, :]
    theta = np.arctan(alpha[1] / alpha[0])
    if alpha[0] < 0:
        theta = np.pi + theta
    
    Rot = np.zeros_like(Proc_rot)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    for i in range(n):
        Rot[:, :, i] = Proc_rot[:, :, i] @ R
    
    return Rot


def acc2(tb):
    """Classification performance metrics from 2x2 confusion matrix"""
    acc = np.sum(np.diag(tb)) / np.sum(tb)
    sens = tb[0, 0] / np.sum(tb[:, 0]) if np.sum(tb[:, 0]) > 0 else 0
    spec = tb[1, 1] / np.sum(tb[:, 1]) if np.sum(tb[:, 1]) > 0 else 0
    
    return {'acc': round(acc, 4), 'sens': round(sens, 4), 'spec': round(spec, 4)}


def summary_results_is(class_results):
    """Summary of in-sample classification results"""
    print("LDA")
    tr = pd.crosstab(class_results['predicted_class_LDA']['pred_class'],
                     class_results['predicted_class_LDA']['true_class'])
    print(tr)
    print(acc2(tr.values))
    
    print("\nLR")
    tr = pd.crosstab(class_results['predicted_class_LR']['pred_class'],
                     class_results['predicted_class_LR']['true_class'])
    print(tr)
    print(acc2(tr.values))
    
    print("\nkNN")
    tr = pd.crosstab(class_results['predicted_class_knn']['pred_class'],
                     class_results['predicted_class_knn']['true_class'])
    print(tr)
    print(acc2(tr.values))


def summary_results_oos(class_results):
    """Summary of out-of-sample classification results"""
    print("LDA mean alignment")
    tr = pd.crosstab(class_results['predicted_class_LDA']['mean_registration'],
                     class_results['predicted_class_LDA']['true_class'])
    print(tr)
    print(acc2(tr.values))
    
    print("\nLDA median alignment")
    tr = pd.crosstab(class_results['predicted_class_LDA']['median_registration'],
                     class_results['predicted_class_LDA']['true_class'])
    print(tr)
    print(acc2(tr.values))
    
    print("\nLR mean alignment")
    tr = pd.crosstab(class_results['predicted_class_LR']['mean_registration'],
                     class_results['predicted_class_LR']['true_class'])
    print(tr)
    print(acc2(tr.values))
    
    print("\nLR median alignment")
    tr = pd.crosstab(class_results['predicted_class_LR']['median_registration'],
                     class_results['predicted_class_LR']['true_class'])
    print(tr)
    print(acc2(tr.values))
    
    print("\nk-NN mean alignment")
    tr = pd.crosstab(class_results['predicted_class_knn']['mean_registration'],
                     class_results['predicted_class_knn']['true_class'])
    print(tr)
    print(acc2(tr.values))
    
    print("\nk-NN median alignment")
    tr = pd.crosstab(class_results['predicted_class_knn']['median_registration'],
                     class_results['predicted_class_knn']['true_class'])
    print(tr)
    print(acc2(tr.values))


def table2(x, y):
    """Create 2x2 table even when all individuals are classified in the same group"""
    ta = pd.crosstab(x, y)
    
    if ta.shape[0] < 2 or ta.shape[1] < 2:
        R = pd.DataFrame(0, index=['G1', 'G2'], columns=['G1', 'G2'])
        for i in ta.index:
            for j in ta.columns:
                R.loc[i, j] = ta.loc[i, j]
        ta = R.values
    else:
        ta = ta.values
    
    return ta


def append_tables_oos(class_results, tables):
    """Append out-of-sample classification results"""
    tables[0] = tables[0] + table2(class_results['predicted_class_LDA']['mean_registration'],
                                    class_results['predicted_class_LDA']['true_class'])
    tables[1] = tables[1] + table2(class_results['predicted_class_LDA']['median_registration'],
                                    class_results['predicted_class_LDA']['true_class'])
    tables[2] = tables[2] + table2(class_results['predicted_class_LR']['mean_registration'],
                                    class_results['predicted_class_LR']['true_class'])
    tables[3] = tables[3] + table2(class_results['predicted_class_LR']['median_registration'],
                                    class_results['predicted_class_LR']['true_class'])
    tables[4] = tables[4] + table2(class_results['predicted_class_knn']['mean_registration'],
                                    class_results['predicted_class_knn']['true_class'])
    tables[5] = tables[5] + table2(class_results['predicted_class_knn']['median_registration'],
                                    class_results['predicted_class_knn']['true_class'])
    
    return tables


def acc2_list(tables, n_runs):
    """Classification performance metrics from list of confusion matrices"""
    res = []
    for i in range(len(tables)):
        res.append(acc2(tables[i] / n_runs))
    return res
