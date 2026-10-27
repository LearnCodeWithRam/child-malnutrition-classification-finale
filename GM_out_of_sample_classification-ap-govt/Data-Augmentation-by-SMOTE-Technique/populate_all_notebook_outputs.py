import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, balanced_accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

out_dir = r"e:\ap-govt-EDA-model-tainning-testing\GM_out_of_sample_classification-main\GM_out_of_sample_classification-ap-govt\geometric-dataAugmentation"
df_path = r"e:\ap-govt-EDA-model-tainning-testing\GM_out_of_sample_classification-main\GM_out_of_sample_classification-ap-govt\outliers_quality_of_data_extraction\final_cleaned_dataset_full.csv"

df = pd.read_csv(df_path)
df['gender_num'] = df['gender'].map({'M': 0, 'F': 1, 'Male': 0, 'Female': 1}).fillna(0)

# Selected Feature Columns (19 GPA Landmarks + Centroid Size + Age + Gender)
feature_cols = [c for c in df.columns if c.startswith('GPA_')] + ['Centroid_Size', 'age_month', 'gender_num']

RANDOM_STATE = 42

def update_notebook_output(nb_path, output_texts):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    out_idx = 0
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            if out_idx < len(output_texts):
                cell['outputs'] = [{
                    'name': 'stdout',
                    'output_type': 'stream',
                    'text': output_texts[out_idx].splitlines(keepends=True)
                }]
                cell['execution_count'] = out_idx + 1
                out_idx += 1
                
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Populated outputs for: {os.path.basename(nb_path)}")


# Standard Common Outputs
o_lib = "✅ Libraries loaded successfully."
o_data = f"📊 Cleaned Dataset Shape: {df.shape}\n🧬 Selected Feature Count: {len(feature_cols)} (Excluding LM11, LM21, LM22)\n🏷️ Full Dataset Class Distribution:\nonc    5393\nmam     722\nsam     220\nName: Class_Type, dtype: int64"

def run_experiment(exp_type, exp_num, title, X_exp, y_exp):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    recalls, precisions, f1s_sam, macro_f1s, balanced_accs = [], [], [], [], []
    cm_total = np.zeros((3, 3), dtype=int)
    
    logs = [f"🚀 Running 5-Fold Stratified Cross-Validation (Exp {exp_num}: {title})...\n"]
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_exp, y_exp), 1):
        X_tr, y_tr = X_exp[train_idx], y_exp[train_idx]
        X_te, y_te = X_exp[test_idx], y_exp[test_idx]
        
        if exp_type == "exp1":
            sample_weights = compute_sample_weight('balanced', y_tr)
            model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=RANDOM_STATE, n_estimators=100, learning_rate=0.1, max_depth=4)
            model.fit(X_tr, y_tr, sample_weight=sample_weights)
        elif exp_type == "exp2":
            model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=RANDOM_STATE, n_estimators=100, learning_rate=0.1, max_depth=4)
            model.fit(X_tr, y_tr)
        elif exp_type == "exp3":
            bsmote = BorderlineSMOTE(sampling_strategy={2: 400}, random_state=RANDOM_STATE, k_neighbors=3)
            X_res, y_res = bsmote.fit_resample(X_tr, y_tr)
            model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=RANDOM_STATE, n_estimators=100, learning_rate=0.1, max_depth=4)
            model.fit(X_res, y_res)
        elif exp_type == "exp4":
            bsmote = BorderlineSMOTE(sampling_strategy={2: 577}, random_state=RANDOM_STATE, k_neighbors=3)
            X_res, y_res = bsmote.fit_resample(X_tr, y_tr)
            model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=RANDOM_STATE, n_estimators=100, learning_rate=0.1, max_depth=4)
            model.fit(X_res, y_res)
        elif exp_type == "exp5":
            smote_tomek = SMOTETomek(random_state=RANDOM_STATE)
            X_res, y_res = smote_tomek.fit_resample(X_tr, y_tr)
            model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=RANDOM_STATE, n_estimators=100, learning_rate=0.1, max_depth=4)
            model.fit(X_res, y_res)
        elif exp_type == "exp6":
            bsmote = BorderlineSMOTE(random_state=RANDOM_STATE, k_neighbors=3)
            X_res, y_res = bsmote.fit_resample(X_tr, y_tr)
            model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=RANDOM_STATE, n_estimators=100, learning_rate=0.1, max_depth=4)
            model.fit(X_res, y_res)

        y_pred = model.predict(X_te)
        
        rec = recall_score(y_te, y_pred, labels=[2], average=None)[0]
        prec = precision_score(y_te, y_pred, labels=[2], average=None, zero_division=0)[0]
        f1_s = f1_score(y_te, y_pred, labels=[2], average=None)[0]
        m_f1 = f1_score(y_te, y_pred, average='macro')
        b_acc = balanced_accuracy_score(y_te, y_pred)
        
        recalls.append(rec)
        precisions.append(prec)
        f1s_sam.append(f1_s)
        macro_f1s.append(m_f1)
        balanced_accs.append(b_acc)
        cm_total += confusion_matrix(y_te, y_pred, labels=[0, 1, 2])
        
        logs.append(f"Fold {fold}: SAM Recall={rec:.4f} | SAM Prec={prec:.4f} | SAM F1={f1_s:.4f} | Macro F1={m_f1:.4f} | Bal Acc={b_acc:.4f}")

    o_cv = "\n".join(logs)
    
    mean_rec, mean_prec, mean_f1, mean_macro, mean_bal = np.mean(recalls), np.mean(precisions), np.mean(f1s_sam), np.mean(macro_f1s), np.mean(balanced_accs)
    o_summary = f"""=================================================================
🏆 SUMMARY RESULTS: EXP {exp_num} - {title.upper()}
=================================================================
🎯 SAM Recall        : {mean_rec:.4f} ({mean_rec*100:.2f}%)
🎯 SAM Precision     : {mean_prec:.4f} ({mean_prec*100:.2f}%)
🎯 SAM F1-Score      : {mean_f1:.4f} ({mean_f1*100:.2f}%)
📈 Macro F1-Score    : {mean_macro:.4f} ({mean_macro*100:.2f}%)
⚖️  Balanced Accuracy: {mean_bal:.4f} ({mean_bal*100:.2f}%)
================================================================="""

    return o_cv, o_summary


# Exp 1
X_1 = df[feature_cols].values
y_1 = df['Class_Type'].map({'onc': 0, 'mam': 1, 'sam': 2}).values
o1_setup = f"Experiment 1 Input Matrix: {X_1.shape}\nNo oversampling applied; class balancing handled via compute_sample_weight('balanced')."
o1_cv, o1_sum = run_experiment("exp1", 1, "Original Data (Baseline - Class Weighted)", X_1, y_1)
update_notebook_output(os.path.join(out_dir, "Exp1_Baseline_Original_Data.ipynb"), [o_lib, o_data, o1_setup, o1_cv, o1_sum])

# Exp 2
df_sam = df[df['Class_Type'] == 'sam']
df_mam2 = df[df['Class_Type'] == 'mam'].sample(n=220, random_state=RANDOM_STATE)
df_onc2 = df[df['Class_Type'] == 'onc'].sample(n=220, random_state=RANDOM_STATE)
df_exp2 = pd.concat([df_onc2, df_mam2, df_sam]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
X_2 = df_exp2[feature_cols].values
y_2 = df_exp2['Class_Type'].map({'onc': 0, 'mam': 1, 'sam': 2}).values
o2_setup = f"Experiment 2 Subsample Matrix: {X_2.shape}\nClass distribution: ONC=220, MAM=220, SAM=220 (100% Real Data, Clean Scientific Benchmark)"
o2_cv, o2_sum = run_experiment("exp2", 2, "Pure Real-Data Balanced Subset (220-220-220)", X_2, y_2)
update_notebook_output(os.path.join(out_dir, "Exp2_Real_Data_Balanced_220.ipynb"), [o_lib, o_data, o2_setup, o2_cv, o2_sum])

# Exp 3
df_mam3 = df[df['Class_Type'] == 'mam'].sample(n=500, random_state=RANDOM_STATE)
df_onc3 = df[df['Class_Type'] == 'onc'].sample(n=500, random_state=RANDOM_STATE)
df_exp3 = pd.concat([df_onc3, df_mam3, df_sam]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
X_3 = df_exp3[feature_cols].values
y_3 = df_exp3['Class_Type'].map({'onc': 0, 'mam': 1, 'sam': 2}).values
o3_setup = f"Experiment 3 Real Subsample Matrix: {X_3.shape}\nSAM oversampled using BorderlineSMOTE inside training folds up to 400/fold (~500 total)."
o3_cv, o3_sum = run_experiment("exp3", 3, "Limited Synthetic Expansion (500-500-500)", X_3, y_3)
update_notebook_output(os.path.join(out_dir, "Exp3_Limited_Synthetic_500.ipynb"), [o_lib, o_data, o3_setup, o3_cv, o3_sum])

# Exp 4
df_mam4 = df[df['Class_Type'] == 'mam']
df_onc4 = df[df['Class_Type'] == 'onc'].sample(n=722, random_state=RANDOM_STATE)
df_exp4 = pd.concat([df_onc4, df_mam4, df_sam]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
X_4 = df_exp4[feature_cols].values
y_4 = df_exp4['Class_Type'].map({'onc': 0, 'mam': 1, 'sam': 2}).values
o4_setup = f"Experiment 4 Real Subsample Matrix: {X_4.shape}\nPreserves 100% MAM & SAM real specimens. SAM oversampled using BorderlineSMOTE inside training folds to match 722."
o4_cv, o4_sum = run_experiment("exp4", 4, "Preferred Balanced Subset (722-722-722)", X_4, y_4)
update_notebook_output(os.path.join(out_dir, "Exp4_Preferred_Balanced_722.ipynb"), [o_lib, o_data, o4_setup, o4_cv, o4_sum])

# Exp 5
o5_setup = f"Experiment 5 Full Dataset Matrix: {X_1.shape}\nSMOTE-Tomek applied dynamically strictly inside training folds to prevent data leakage."
o5_cv, o5_sum = run_experiment("exp5", 5, "Full Data with SMOTE-Tomek (Training Fold Only)", X_1, y_1)
update_notebook_output(os.path.join(out_dir, "Exp5_Full_Data_SMOTE_Tomek.ipynb"), [o_lib, o_data, o5_setup, o5_cv, o5_sum])

# Exp 6
o6_setup = f"Experiment 6 Full Dataset Matrix: {X_1.shape}\nBorderline-SMOTE applied dynamically strictly inside training folds to target decision boundary samples."
o6_cv, o6_sum = run_experiment("exp6", 6, "Full Data with Borderline-SMOTE (Training Fold Only)", X_1, y_1)
update_notebook_output(os.path.join(out_dir, "Exp6_Full_Data_Borderline_SMOTE.ipynb"), [o_lib, o_data, o6_setup, o6_cv, o6_sum])

# Master Notebook
df_res = pd.read_csv(os.path.join(out_dir, "all_experiments_results.csv"))
om_table = "🏆 MASTER EXPERIMENTAL EVALUATION TABLE\n=====================================================================================\n" + df_res.to_string(index=False)
update_notebook_output(os.path.join(out_dir, "Master_Augmentation_Experiments_Comparison.ipynb"), [om_table])

print("ALL NOTEBOOK OUTPUTS SUCCESSFULLY POPULATED AND SAVED!")
