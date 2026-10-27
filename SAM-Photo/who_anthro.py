"""
WHO Child Growth Standards (Anthro) Python Implementation
Ported directly from the official WHO R package:
  'anthro' (https://github.com/worldhealthorganization/anthro)
References:
  - WHO Child Growth Standards: Length/height-for-age, weight-for-age,
    weight-for-length, weight-for-height and body mass index-for-age (WHO, 2006)
  - WHO Child Growth Standards: Head circumference-for-age, arm circumference-for-age,
    triceps skinfold-for-age and subscapular skinfold-for-age (WHO, 2007)
  - http://www.who.int/childgrowth/standards/Chap_7.pdf
"""

import os
import numpy as np
import pandas as pd

# Constants
ANTHRO_DAYS_OF_MONTH = 30.4375

# Path to LMS growth standards tables
TABLES_DIR = os.path.join(
    os.path.dirname(__file__),
    'anthro-master', 'anthro-master', 'data-raw', 'growthstandards'
)

def load_lms_table(filename):
    path = os.path.join(TABLES_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"LMS table not found: {path}")
    return pd.read_csv(path, sep='\t')

# Global cache of tables
_TABLES = {}

def get_table(name):
    if name not in _TABLES:
        _TABLES[name] = load_lms_table(f"{name}.txt")
    return _TABLES[name]

def compute_zscore(y, m, l, s):
    """
    Standard non-adjusted LMS Z-score formula:
    Z = ((y / m)^l - 1) / (s * l)
    When l == 0: Z = ln(y / m) / s
    """
    y = np.asarray(y, dtype=float)
    m = np.asarray(m, dtype=float)
    l = np.asarray(l, dtype=float)
    s = np.asarray(s, dtype=float)
    
    with np.errstate(all='ignore'):
        cond_zero = np.isclose(l, 0)
        res = np.where(
            cond_zero,
            np.log(y / m) / s,
            ((y / m)**l - 1.0) / (s * l)
        )
        res = np.where((y <= 0) | (m <= 0) | (s <= 0), np.nan, res)
    return res

def compute_zscore_adjusted(y, m, l, s):
    """
    WHO Adjusted Z-score for right/left-skewed anthropometric indicators:
    For |Z| > 3, applies the restricted linear extrapolation beyond +/- 3 SD:
    SD3pos = m * (1 + l * s * 3)^(1 / l)
    SD3neg = m * (1 + l * s * (-3))^(1 / l)
    SD23pos = SD3pos - m * (1 + l * s * 2)^(1 / l)
    SD23neg = m * (1 + l * s * (-2))^(1 / l) - SD3neg
    If Z > 3:  Z_adj = 3 + (y - SD3pos) / SD23pos
    If Z < -3: Z_adj = -3 + (y - SD3neg) / SD23neg
    """
    y = np.asarray(y, dtype=float)
    m = np.asarray(m, dtype=float)
    l = np.asarray(l, dtype=float)
    s = np.asarray(s, dtype=float)
    
    z = compute_zscore(y, m, l, s)
    
    with np.errstate(all='ignore'):
        def calc_sd(sd):
            return m * ((1.0 + l * s * sd)**(1.0 / l))
        
        sd3pos = calc_sd(3.0)
        sd3neg = calc_sd(-3.0)
        sd23pos = sd3pos - calc_sd(2.0)
        sd23neg = calc_sd(-2.0) - sd3neg
        
        z_adj = z.copy()
        
        mask_gt3 = ~np.isnan(z) & (z > 3.0)
        z_adj[mask_gt3] = 3.0 + (y[mask_gt3] - sd3pos[mask_gt3]) / sd23pos[mask_gt3]
        
        mask_lt3 = ~np.isnan(z) & (z < -3.0)
        z_adj[mask_lt3] = -3.0 + (y[mask_lt3] - sd3neg[mask_lt3]) / sd23neg[mask_lt3]
        
    return z_adj

def standardize_sex(sex):
    """
    Standardize sex: 1 for male, 2 for female.
    """
    if isinstance(sex, (pd.Series, np.ndarray, list)):
        s_str = pd.Series(sex).astype(str).str.strip().str.lower()
        res = pd.Series(np.nan, index=s_str.index, dtype=float)
        res[s_str.isin(['1', 'm', 'male'])] = 1.0
        res[s_str.isin(['2', 'f', 'female'])] = 2.0
        return res.values
    else:
        s = str(sex).strip().lower()
        if s in ['1', 'm', 'male']:
            return 1
        elif s in ['2', 'f', 'female']:
            return 2
        return np.nan

def adjust_lenhei(age_in_days, lenhei, measure=None):
    """
    Standardize length/height according to WHO protocol:
    If age < 731 days (24 months) and measured as standing height ('h'), add 0.7 cm to get recumbent length.
    If age >= 731 days and measured as recumbent length ('l'), subtract 0.7 cm to get standing height.
    If measure is None:
       assume 'l' for age < 731 days, and 'h' for age >= 731 days (no change needed).
    """
    age_in_days = np.asarray(age_in_days, dtype=float)
    lenhei = np.asarray(lenhei, dtype=float).copy()
    
    if measure is None:
        return lenhei
    
    meas_series = pd.Series(measure).astype(str).str.strip().str.lower()
    
    # under 24m measured as standing height -> add 0.7cm
    mask_add = (age_in_days < 731) & (meas_series == 'h')
    lenhei[mask_add] += 0.7
    
    # >= 24m measured as recumbent length -> subtract 0.7cm
    mask_sub = (age_in_days >= 731) & (meas_series == 'l')
    lenhei[mask_sub] -= 0.7
    
    return lenhei

def anthro_zscore_length_for_age(lenhei, age_in_days, sex):
    """
    Length/Height-for-age Z-score (HAZ / Stunting). Uses lenanthro.txt table.
    """
    df_lms = get_table('lenanthro')
    n = len(lenhei)
    z_scores = np.full(n, np.nan)
    
    inp = pd.DataFrame({
        'measure': lenhei,
        'age_in_days': np.round(age_in_days).astype(int),
        'sex': sex,
        'order': np.arange(n)
    })
    
    merged = inp.merge(df_lms, left_on=['age_in_days', 'sex'], right_on=['age', 'sex'], how='left')
    merged = merged.sort_values('order')
    
    valid = (merged['age_in_days'] >= 0) & (merged['age_in_days'] <= 1856) & (merged['measure'] > 0)
    y = merged.loc[valid, 'measure'].values
    m = merged.loc[valid, 'm'].values
    l = merged.loc[valid, 'l'].values
    s = merged.loc[valid, 's'].values
    
    z_calc = compute_zscore(y, m, l, s)
    z_scores[merged.loc[valid, 'order'].values] = np.round(z_calc, 2)
    return z_scores

def anthro_zscore_weight_for_age(weight, age_in_days, sex, oedema=None):
    """
    Weight-for-age Z-score (WAZ / Underweight). Uses weianthro.txt table.
    """
    df_lms = get_table('weianthro')
    n = len(weight)
    z_scores = np.full(n, np.nan)
    
    inp = pd.DataFrame({
        'measure': weight,
        'age_in_days': np.round(age_in_days).astype(int),
        'sex': sex,
        'order': np.arange(n)
    })
    
    merged = inp.merge(df_lms, left_on=['age_in_days', 'sex'], right_on=['age', 'sex'], how='left')
    merged = merged.sort_values('order')
    
    valid = (merged['age_in_days'] >= 0) & (merged['age_in_days'] <= 1856) & (merged['measure'] > 0)
    if oedema is not None:
        oed_s = pd.Series(oedema).astype(str).str.strip().str.lower()
        valid = valid & ~oed_s.isin(['y', '1'])
        
    y = merged.loc[valid, 'measure'].values
    m = merged.loc[valid, 'm'].values
    l = merged.loc[valid, 'l'].values
    s = merged.loc[valid, 's'].values
    
    z_calc = compute_zscore_adjusted(y, m, l, s)
    z_scores[merged.loc[valid, 'order'].values] = np.round(z_calc, 2)
    return z_scores

def anthro_zscore_weight_for_lenhei(weight, lenhei, age_in_days, sex, oedema=None):
    """
    Weight-for-length/height Z-score (WHZ / WFH / Wasting).
    Uses wflanthro.txt (length 45-110 cm) for age < 731 days,
    and wfhanthro.txt (height 65-120 cm) for age >= 731 days.
    Linearly interpolates LMS parameters between adjacent 0.1 cm steps.
    """
    df_wfl = get_table('wflanthro').copy()
    df_wfl.columns = ['sex', 'lenhei', 'l', 'm', 's', 'lorh']
    df_wfl['lorh'] = df_wfl['lorh'].str.lower()
    
    df_wfh = get_table('wfhanthro').copy()
    df_wfh.columns = ['sex', 'lenhei', 'l', 'm', 's', 'lorh']
    df_wfh['lorh'] = df_wfh['lorh'].str.lower()
    
    df_standards = pd.concat([df_wfl, df_wfh], ignore_index=True)
    
    n = len(weight)
    z_scores = np.full(n, np.nan)
    
    # Join on 'l' (length) if age < 731 days or lenhei < 87 when age is missing
    join_on_l = (age_in_days < 731) | (np.isnan(age_in_days) & (lenhei < 87))
    join_col = np.where(join_on_l, 'l', 'h')
    
    low_lenhei = np.trunc(lenhei * 10.0) / 10.0
    upp_lenhei = np.trunc(lenhei * 10.0 + 1.0) / 10.0
    diff_lenhei = (lenhei - low_lenhei) / 0.1
    
    inp = pd.DataFrame({
        'weight': weight,
        'lenhei': lenhei,
        'sex': sex,
        'join_col': join_col,
        'low_lenhei': low_lenhei,
        'upp_lenhei': upp_lenhei,
        'diff_lenhei': diff_lenhei,
        'age_in_days': age_in_days,
        'order': np.arange(n)
    })
    
    # Merge lower bounds
    m1 = inp.merge(
        df_standards,
        left_on=['sex', 'low_lenhei', 'join_col'],
        right_on=['sex', 'lenhei', 'lorh'],
        how='left'
    )
    
    # Merge upper bounds
    m2 = m1.merge(
        df_standards,
        left_on=['sex', 'upp_lenhei', 'join_col'],
        right_on=['sex', 'lenhei', 'lorh'],
        how='left',
        suffixes=('', '_upper')
    )
    m2 = m2.sort_values('order')
    
    # Valid filter:
    # Length valid range: 45 to 110 cm
    # Height valid range: 65 to 120 cm
    valid_range = np.where(
        m2['join_col'] == 'l',
        (m2['lenhei'] >= 45.0) & (m2['lenhei'] <= 110.0),
        (m2['lenhei'] >= 65.0) & (m2['lenhei'] <= 120.0)
    )
    valid = valid_range & (m2['weight'] > 0) & (m2['age_in_days'] <= 1856) & ~m2['m'].isna()
    
    if oedema is not None:
        oed_s = pd.Series(oedema).astype(str).str.strip().str.lower()
        valid = valid & ~oed_s.isin(['y', '1'])
        
    diff = m2.loc[valid, 'diff_lenhei'].values
    y = m2.loc[valid, 'weight'].values
    
    # Linear interpolation of LMS parameters
    m_low = m2.loc[valid, 'm'].values
    m_upp = m2.loc[valid, 'm_upper'].values
    m = np.where(diff > 0, m_low + diff * (m_upp - m_low), m_low)
    
    l_low = m2.loc[valid, 'l'].values
    l_upp = m2.loc[valid, 'l_upper'].values
    l = np.where(diff > 0, l_low + diff * (l_upp - l_low), l_low)
    
    s_low = m2.loc[valid, 's'].values
    s_upp = m2.loc[valid, 's_upper'].values
    s = np.where(diff > 0, s_low + diff * (s_upp - s_low), s_low)
    
    z_calc = compute_zscore_adjusted(y, m, l, s)
    z_scores[m2.loc[valid, 'order'].values] = np.round(z_calc, 2)
    return z_scores

def anthro_zscore_arm_circumference_for_age(armc, age_in_days, sex):
    """
    Arm circumference-for-age Z-score (ACAZ / ACA). Uses acanthro.txt table.
    Valid for age 91 to 1856 days (3 to 60 months).
    """
    df_lms = get_table('acanthro')
    n = len(armc)
    z_scores = np.full(n, np.nan)
    
    inp = pd.DataFrame({
        'measure': armc,
        'age_in_days': np.round(age_in_days).astype(int),
        'sex': sex,
        'order': np.arange(n)
    })
    
    merged = inp.merge(df_lms, left_on=['age_in_days', 'sex'], right_on=['age', 'sex'], how='left')
    merged = merged.sort_values('order')
    
    valid = (merged['age_in_days'] >= 91) & (merged['age_in_days'] <= 1856) & (merged['measure'] > 0) & ~merged['m'].isna()
    y = merged.loc[valid, 'measure'].values
    m = merged.loc[valid, 'm'].values
    l = merged.loc[valid, 'l'].values
    s = merged.loc[valid, 's'].values
    
    z_calc = compute_zscore_adjusted(y, m, l, s)
    z_scores[merged.loc[valid, 'order'].values] = np.round(z_calc, 2)
    return z_scores

def anthro_zscore_bmi_for_age(bmi, age_in_days, sex, oedema=None):
    """
    BMI-for-age Z-score (BAZ / BIA). Uses bmianthro.txt table.
    """
    df_lms = get_table('bmianthro')
    n = len(bmi)
    z_scores = np.full(n, np.nan)
    
    inp = pd.DataFrame({
        'measure': bmi,
        'age_in_days': np.round(age_in_days).astype(int),
        'sex': sex,
        'order': np.arange(n)
    })
    
    merged = inp.merge(df_lms, left_on=['age_in_days', 'sex'], right_on=['age', 'sex'], how='left')
    merged = merged.sort_values('order')
    
    valid = (merged['age_in_days'] >= 0) & (merged['age_in_days'] <= 1856) & (merged['measure'] > 0) & ~merged['m'].isna()
    if oedema is not None:
        oed_s = pd.Series(oedema).astype(str).str.strip().str.lower()
        valid = valid & ~oed_s.isin(['y', '1'])
        
    y = merged.loc[valid, 'measure'].values
    m = merged.loc[valid, 'm'].values
    l = merged.loc[valid, 'l'].values
    s = merged.loc[valid, 's'].values
    
    z_calc = compute_zscore_adjusted(y, m, l, s)
    z_scores[merged.loc[valid, 'order'].values] = np.round(z_calc, 2)
    return z_scores

def compute_all_who_anthro(df, age_col='age_month', sex_col='gender',
                          weight_col='weight_kg', height_col='height_cm',
                          muac_col='muac_cm', oedema_col=None):
    """
    Compute complete WHO Anthropometric Z-scores and classifications for a dataframe.
    """
    res = pd.DataFrame(index=df.index)
    
    # Standardize age in days
    age_m = df[age_col].values.astype(float)
    age_d = np.round(age_m * ANTHRO_DAYS_OF_MONTH)
    
    # Standardize sex
    csex = standardize_sex(df[sex_col])
    
    # Clean weight and height
    weight = df[weight_col].values.astype(float)
    lenhei = df[height_col].values.astype(float)
    
    # Standardize length/height
    clenhei = adjust_lenhei(age_d, lenhei)
    
    # Clean MUAC (standardize mm to cm if values > 40)
    armc = df[muac_col].values.astype(float)
    armc = np.where(armc > 40, armc / 10.0, armc)
    
    # BMI
    cbmi = weight / ((clenhei / 100.0)**2)
    
    oedema = df[oedema_col].values if oedema_col and oedema_col in df.columns else None
    
    # Compute WHO Z-scores
    res['who_zlen'] = anthro_zscore_length_for_age(clenhei, age_d, csex)
    res['who_zwei'] = anthro_zscore_weight_for_age(weight, age_d, csex, oedema)
    res['who_zwfl'] = anthro_zscore_weight_for_lenhei(weight, clenhei, age_d, csex, oedema)
    res['who_zbmi'] = anthro_zscore_bmi_for_age(cbmi, age_d, csex, oedema)
    res['who_zac']  = anthro_zscore_arm_circumference_for_age(armc, age_d, csex)
    
    # Flag extreme biological implausibilities (|Z| > threshold)
    res['who_flen'] = (np.abs(res['who_zlen']) > 6).astype(int)
    res['who_fwei'] = ((res['who_zwei'] < -6) | (res['who_zwei'] > 5)).astype(int)
    res['who_fwfl'] = (np.abs(res['who_zwfl']) > 5).astype(int)
    res['who_fbmi'] = (np.abs(res['who_zbmi']) > 5).astype(int)
    res['who_fac']  = (np.abs(res['who_zac']) > 5).astype(int)
    
    # WHO Categorical Classifications
    # 1. Wasting by Weight-for-Height (WHZ)
    zwfl = res['who_zwfl'].values
    res['who_wasting_whz'] = np.where(
        np.isnan(zwfl), 'Unknown',
        np.where(zwfl < -3.0, 'SAM',
        np.where(zwfl < -2.0, 'MAM', 'Normal'))
    )
    
    # 2. Wasting by Physical MUAC Cutoff (WHO Clinical Guideline for 6-59 months)
    res['who_wasting_muac'] = np.where(
        np.isnan(armc) | (armc <= 0), 'Unknown',
        np.where(armc < 11.5, 'SAM',
        np.where(armc < 12.5, 'MAM', 'Normal'))
    )
    
    # 3. Wasting by MUAC-for-Age Z-score (ACAZ)
    zac = res['who_zac'].values
    res['who_wasting_acaz'] = np.where(
        np.isnan(zac), 'Unknown',
        np.where(zac < -3.0, 'SAM',
        np.where(zac < -2.0, 'MAM', 'Normal'))
    )
    
    # 4. Stunting by Height-for-Age (HAZ)
    zlen = res['who_zlen'].values
    res['who_stunting_haz'] = np.where(
        np.isnan(zlen), 'Unknown',
        np.where(zlen < -3.0, 'Severely Stunted',
        np.where(zlen < -2.0, 'Moderately Stunted', 'Normal'))
    )
    
    # 5. Underweight by Weight-for-Age (WAZ)
    zwei = res['who_zwei'].values
    res['who_underweight_waz'] = np.where(
        np.isnan(zwei), 'Unknown',
        np.where(zwei < -3.0, 'Severely Underweight',
        np.where(zwei < -2.0, 'Moderately Underweight', 'Normal'))
    )
    
    return res
