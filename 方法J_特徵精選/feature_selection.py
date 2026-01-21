"""
方法 J: 特徵精選
從所有方法 (B, G, H, I) 的特徵中精選最重要的 45 個

策略:
1. 整合所有曾建立的特徵 (150+)
2. 用特徵重要性初篩到 80 個
3. 用 RFE 精選到 45 個
4. 用方法 E 的最佳超參數訓練
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.metrics import f1_score, accuracy_score, classification_report
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 載入原始資料
# ============================================================
print("=" * 70)
print("方法 J: 特徵精選 (從 150+ 個特徵精選 45 個)")
print("=" * 70)

train_df = pd.read_csv('../資料清洗/train_cleaned.csv')
test_df = pd.read_csv('../資料清洗/test_cleaned.csv')

print(f"\n原始訓練集: {train_df.shape}")
print(f"原始測試集: {test_df.shape}")

# ============================================================
# 2. 重新建立所有特徵 (整合 B, G, H, I)
# ============================================================
print("\n\n【整合所有特徵】")
print("=" * 70)

# --- 2.1 基礎特徵 (方法 B) ---
print("\n[1] 方法 B: 基礎特徵工程...")

def extract_lab_features(lab_str):
    """從 Lab_Values 提取特徵"""
    features = {
        'lab_count': 0, 'lab_abnormal_H': 0, 'lab_abnormal_L': 0,
        'lab_critical_HH': 0, 'lab_critical_LL': 0, 'lab_has_data': 0,
    }
    if pd.isna(lab_str):
        return features
    features['lab_has_data'] = 1
    pattern = r'\(([HLN]+)\)'
    matches = re.findall(pattern, str(lab_str))
    features['lab_count'] = len(matches)
    for status in matches:
        if status == 'HH': features['lab_critical_HH'] += 1
        elif status == 'LL': features['lab_critical_LL'] += 1
        elif status == 'H': features['lab_abnormal_H'] += 1
        elif status == 'L': features['lab_abnormal_L'] += 1
    return features

def extract_med_features(row):
    """從 Medication 提取特徵"""
    features = {
        'med_count': 0, 'has_chronic_med': 0, 'has_cardiac_med': 0,
        'has_diabetes_med': 0, 'has_anticoagulant': 0, 'has_opioid': 0, 'has_sedative': 0,
    }
    med_str = str(row.get('Medication_Usage', ''))
    drug_cat = str(row.get('Drug_Category', ''))
    drug_name = str(row.get('Drug_Standardized', '')).lower()

    if pd.notna(row.get('Medication_Usage')) and med_str != 'nan':
        features['med_count'] = len(med_str.split(','))
    chronic_categories = ['ANTIHYPERTENSIVES', 'CARDIAC', 'DIURETICS', 'ANTIARRHYTHMICS']
    if drug_cat in chronic_categories:
        features['has_chronic_med'] = 1
        features['has_cardiac_med'] = 1
    if drug_cat == 'INSULIN' or 'metformin' in drug_name or 'glipizide' in drug_name:
        features['has_diabetes_med'] = 1
        features['has_chronic_med'] = 1
    if drug_cat == 'ANTICOAGULANTS': features['has_anticoagulant'] = 1
    if drug_cat == 'OPIOID_ANALGESICS': features['has_opioid'] = 1
    if drug_cat == 'SEDATIVES_HYPNOTICS': features['has_sedative'] = 1
    return features

def extract_catheter_features(cath_str):
    """從 Catheter_Use 提取特徵"""
    features = {
        'catheter_count': 0, 'has_catheter': 0, 'has_piv': 0, 'has_urinary': 0,
        'has_cvc': 0, 'has_arterial': 0, 'has_chest_tube': 0, 'has_wound': 0,
    }
    if pd.isna(cath_str):
        return features
    cath_str = str(cath_str).upper()
    features['has_catheter'] = 1
    features['catheter_count'] = len(cath_str.split(','))
    if 'PIV' in cath_str or 'PERIPHERAL IV' in cath_str: features['has_piv'] = 1
    if 'URINARY' in cath_str: features['has_urinary'] = 1
    if 'CVC' in cath_str or 'CENTRAL' in cath_str or 'PICC' in cath_str: features['has_cvc'] = 1
    if 'ARTERIAL' in cath_str or 'ART' in cath_str: features['has_arterial'] = 1
    if 'CHEST TUBE' in cath_str: features['has_chest_tube'] = 1
    if 'WOUND' in cath_str or 'INCISION' in cath_str or 'DRAIN' in cath_str: features['has_wound'] = 1
    return features

def extract_other_features(row):
    """其他衍生特徵"""
    features = {'age_group': 0, 'bmi_category': 0, 'is_elderly': 0, 'is_obese': 0, 'is_morbid_obese': 0}
    age = row.get('Age', 0)
    if age < 40: features['age_group'] = 0
    elif age < 60: features['age_group'] = 1
    elif age < 75: features['age_group'] = 2
    else: features['age_group'] = 3
    features['is_elderly'] = 1 if age >= 65 else 0
    bmi = row.get('BMI', 0)
    if bmi < 30: features['bmi_category'] = 0
    elif bmi < 35: features['bmi_category'] = 1
    elif bmi < 40: features['bmi_category'] = 2
    else: features['bmi_category'] = 3
    features['is_obese'] = 1 if bmi >= 30 else 0
    features['is_morbid_obese'] = 1 if bmi >= 40 else 0
    return features

# 提取方法 B 特徵
train_lab = train_df['Lab_Values'].apply(extract_lab_features).apply(pd.Series)
test_lab = test_df['Lab_Values'].apply(extract_lab_features).apply(pd.Series)
train_lab['lab_abnormal_total'] = train_lab['lab_abnormal_H'] + train_lab['lab_abnormal_L'] + train_lab['lab_critical_HH'] + train_lab['lab_critical_LL']
test_lab['lab_abnormal_total'] = test_lab['lab_abnormal_H'] + test_lab['lab_abnormal_L'] + test_lab['lab_critical_HH'] + test_lab['lab_critical_LL']

train_med = train_df.apply(extract_med_features, axis=1).apply(pd.Series)
test_med = test_df.apply(extract_med_features, axis=1).apply(pd.Series)

train_cath = train_df['Catheter_Use'].apply(extract_catheter_features).apply(pd.Series)
test_cath = test_df['Catheter_Use'].apply(extract_catheter_features).apply(pd.Series)

train_other = train_df.apply(extract_other_features, axis=1).apply(pd.Series)
test_other = test_df.apply(extract_other_features, axis=1).apply(pd.Series)

print(f"  Lab 特徵: {len(train_lab.columns)}")
print(f"  Med 特徵: {len(train_med.columns)}")
print(f"  Catheter 特徵: {len(train_cath.columns)}")
print(f"  Other 特徵: {len(train_other.columns)}")

# --- 2.2 縱向特徵 (方法 H) ---
print("\n[2] 方法 H: 縱向特徵...")

KEY_TESTS = ['Creatinine', 'Hemoglobin', 'Glucose', 'Sodium', 'Potassium', 'Platelets']

def extract_longitudinal_features(lab_str):
    """提取縱向特徵"""
    features = {}
    for test in KEY_TESTS:
        features[f'h_{test}_count'] = 0
        features[f'h_{test}_abnormal_rate'] = 0
    features['h_total_repeat_count'] = 0
    features['h_has_repeat_tests'] = 0

    if pd.isna(lab_str):
        return features

    lab_str = str(lab_str)
    total_repeats = 0

    for test in KEY_TESTS:
        pattern = rf'{re.escape(test)}[^,]*\[n=(\d+)\]'
        matches = re.findall(pattern, lab_str, re.IGNORECASE)
        if matches:
            count = max(int(m) for m in matches)
            features[f'h_{test}_count'] = count
            total_repeats += count

            status_pattern = rf'{re.escape(test)}[^,]*\(([HLN]+)\)'
            status_matches = re.findall(status_pattern, lab_str, re.IGNORECASE)
            if status_matches:
                abnormal = sum(1 for s in status_matches if s in ['H', 'HH', 'L', 'LL'])
                features[f'h_{test}_abnormal_rate'] = abnormal / len(status_matches)

    features['h_total_repeat_count'] = total_repeats
    features['h_has_repeat_tests'] = 1 if total_repeats > 0 else 0
    return features

train_longitudinal = train_df['Lab_Values'].apply(extract_longitudinal_features).apply(pd.Series)
test_longitudinal = test_df['Lab_Values'].apply(extract_longitudinal_features).apply(pd.Series)
print(f"  縱向特徵: {len(train_longitudinal.columns)}")

# --- 2.3 推測病史特徵 (方法 I) ---
print("\n[3] 方法 I: 推測病史...")

def extract_lab_value(lab_str, test_name):
    """提取特定檢驗值"""
    if pd.isna(lab_str):
        return {'value': None, 'status': None, 'count': 0}
    lab_str = str(lab_str)
    pattern = rf'{re.escape(test_name)}:\s*([\d.]+)\s*[^(]*\(([HLNHLL]+)\)(?:\s*\[n=(\d+)\])?'
    match = re.search(pattern, lab_str, re.IGNORECASE)
    if match:
        return {'value': float(match.group(1)), 'status': match.group(2), 'count': int(match.group(3)) if match.group(3) else 1}
    return {'value': None, 'status': None, 'count': 0}

def infer_medical_history(row):
    """推測病史"""
    lab_str = row.get('Lab_Values', '')
    gender = row.get('Gender', 1)
    history = {
        'hx_ckd': 0, 'hx_diabetes': 0, 'hx_anemia': 0, 'hx_liver': 0,
        'hx_coagulopathy': 0, 'hx_electrolyte': 0, 'hx_cardiac': 0, 'hx_inflammation': 0,
        'hx_count': 0, 'hx_severity': 0,
    }
    if pd.isna(lab_str):
        return history
    lab_str = str(lab_str)

    # CKD
    creatinine = extract_lab_value(lab_str, 'Creatinine')
    if creatinine['value']:
        threshold = 1.2 if gender == 2 else 1.0
        if creatinine['value'] > threshold or creatinine['status'] in ['H', 'HH']:
            history['hx_ckd'] = 1
            history['hx_severity'] += 2 if creatinine['status'] == 'HH' else 1

    # Diabetes
    glucose = extract_lab_value(lab_str, 'Glucose')
    if glucose['value'] and glucose['value'] > 126:
        history['hx_diabetes'] = 1
        history['hx_severity'] += 1
    elif glucose['status'] in ['H', 'HH']:
        history['hx_diabetes'] = 1
        history['hx_severity'] += 2 if glucose['status'] == 'HH' else 1

    # Anemia
    hgb = extract_lab_value(lab_str, 'Hemoglobin')
    if hgb['value']:
        threshold = 13 if gender == 2 else 12
        if hgb['value'] < threshold or hgb['status'] in ['L', 'LL']:
            history['hx_anemia'] = 1
            history['hx_severity'] += 2 if hgb['status'] == 'LL' else 1

    # Liver
    alt = extract_lab_value(lab_str, 'Alanine aminotransferase')
    ast = extract_lab_value(lab_str, 'Aspartate aminotransferase')
    if alt['status'] in ['H', 'HH'] or ast['status'] in ['H', 'HH']:
        history['hx_liver'] = 1
        history['hx_severity'] += 1

    # Coagulopathy
    platelets = extract_lab_value(lab_str, 'Platelets')
    if platelets['status'] in ['L', 'LL']:
        history['hx_coagulopathy'] = 1
        history['hx_severity'] += 2 if platelets['status'] == 'LL' else 1

    # Electrolyte
    sodium = extract_lab_value(lab_str, 'Sodium')
    potassium = extract_lab_value(lab_str, 'Potassium')
    if sodium['status'] in ['H', 'HH', 'L', 'LL'] or potassium['status'] in ['H', 'HH', 'L', 'LL']:
        history['hx_electrolyte'] = 1
        history['hx_severity'] += 1

    # Cardiac
    troponin = extract_lab_value(lab_str, 'Troponin I.cardiac')
    if troponin['status'] in ['H', 'HH']:
        history['hx_cardiac'] = 1
        history['hx_severity'] += 3

    # Inflammation
    wbc = extract_lab_value(lab_str, 'Leukocytes^^corrected for nucleated erythrocytes')
    if wbc['status'] in ['H', 'HH']:
        history['hx_inflammation'] = 1
        history['hx_severity'] += 1

    history['hx_count'] = sum([history['hx_ckd'], history['hx_diabetes'], history['hx_anemia'],
                               history['hx_liver'], history['hx_coagulopathy'], history['hx_electrolyte'],
                               history['hx_cardiac'], history['hx_inflammation']])
    return history

train_history = train_df.apply(infer_medical_history, axis=1).apply(pd.Series)
test_history = test_df.apply(infer_medical_history, axis=1).apply(pd.Series)
print(f"  病史特徵: {len(train_history.columns)}")

# --- 2.4 合併所有基礎特徵 ---
print("\n[4] 合併所有基礎特徵...")

numeric_features = ['Age', 'HEIGHT', 'WEIGHT', 'BMI', 'Surgery_Count']
categorical_features = ['Gender', 'ICU_Patient', 'Anesthesia_Method', 'Patient_Source',
                        'Surgery_Category', 'Drug_Category', 'Route_Standardized', 'Surgery_Procedure_Type']

X_train_base = pd.concat([
    train_df[numeric_features + categorical_features],
    train_lab, train_med, train_cath, train_other,
    train_longitudinal, train_history
], axis=1)

X_test_base = pd.concat([
    test_df[numeric_features + categorical_features],
    test_lab, test_med, test_cath, test_other,
    test_longitudinal, test_history
], axis=1)

y_train = train_df['ASA_Rating'].copy()

print(f"  基礎特徵總數: {len(X_train_base.columns)}")

# ============================================================
# 3. 資料前處理
# ============================================================
print("\n【資料前處理】")

# 處理缺失值
for col in categorical_features:
    if col in X_train_base.columns:
        X_train_base[col] = X_train_base[col].fillna('Unknown')
        X_test_base[col] = X_test_base[col].fillna('Unknown')

numeric_cols = [col for col in X_train_base.columns if col not in categorical_features]
for col in numeric_cols:
    X_train_base[col] = X_train_base[col].fillna(0)
    X_test_base[col] = X_test_base[col].fillna(0)

# Label Encoding
label_encoders = {}
for col in categorical_features:
    if col in X_train_base.columns:
        le = LabelEncoder()
        all_values = pd.concat([X_train_base[col], X_test_base[col]]).unique()
        le.fit(all_values)
        X_train_base[col] = le.transform(X_train_base[col])
        X_test_base[col] = le.transform(X_test_base[col])
        label_encoders[col] = le

# 標準化
scaler = StandardScaler()
X_train_base[numeric_cols] = scaler.fit_transform(X_train_base[numeric_cols])
X_test_base[numeric_cols] = scaler.transform(X_test_base[numeric_cols])

y_train_encoded = y_train - 1

print(f"  前處理後特徵數: {len(X_train_base.columns)}")

# --- 2.5 加入二階交互特徵 ---
print("\n[5] 加入二階交互特徵...")

important_numeric = ['Age', 'BMI', 'Surgery_Count', 'lab_abnormal_total', 'lab_count',
                     'med_count', 'catheter_count', 'hx_count', 'hx_severity',
                     'h_total_repeat_count']
important_numeric = [f for f in important_numeric if f in X_train_base.columns]

X_train_interact = X_train_base[important_numeric].copy()
X_test_interact = X_test_base[important_numeric].copy()

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_interact)
X_test_poly = poly.transform(X_test_interact)

poly_names = poly.get_feature_names_out(important_numeric)
interaction_cols = [col for col in poly_names if ' ' in col]

X_train_poly_df = pd.DataFrame(X_train_poly, columns=poly_names)
X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly_names)

X_train_all = pd.concat([
    X_train_base.reset_index(drop=True),
    X_train_poly_df[interaction_cols].reset_index(drop=True)
], axis=1)

X_test_all = pd.concat([
    X_test_base.reset_index(drop=True),
    X_test_poly_df[interaction_cols].reset_index(drop=True)
], axis=1)

print(f"  交互特徵數: {len(interaction_cols)}")
print(f"  總特徵數: {len(X_train_all.columns)}")

# ============================================================
# 4. 特徵選擇 - 階段 1: 特徵重要性初篩
# ============================================================
print("\n\n【特徵選擇 - 階段 1: 特徵重要性初篩】")
print("=" * 70)

# 用 LightGBM 訓練並獲取特徵重要性
initial_model = lgb.LGBMClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1
)
initial_model.fit(X_train_all, y_train_encoded)

importance_df = pd.DataFrame({
    'Feature': X_train_all.columns,
    'Importance': initial_model.feature_importances_
}).sort_values('Importance', ascending=False)

# 選擇前 80 個
N_STAGE1 = 80
top_features_stage1 = importance_df.head(N_STAGE1)['Feature'].tolist()

print(f"從 {len(X_train_all.columns)} 個特徵初篩到 {N_STAGE1} 個")
print("\nTop 20 特徵:")
for i, (_, row) in enumerate(importance_df.head(20).iterrows()):
    print(f"  {i+1:2}. {row['Feature']:<40} {row['Importance']:.0f}")

X_train_stage1 = X_train_all[top_features_stage1]
X_test_stage1 = X_test_all[top_features_stage1]

# ============================================================
# 5. 特徵選擇 - 階段 2: RFE 精選
# ============================================================
print("\n\n【特徵選擇 - 階段 2: RFE 精選】")
print("=" * 70)

N_FINAL = 45
print(f"從 {N_STAGE1} 個特徵精選到 {N_FINAL} 個...")

# 使用較快的模型進行 RFE
rfe_model = lgb.LGBMClassifier(
    n_estimators=100, max_depth=5, learning_rate=0.1,
    class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1
)

rfe = RFE(estimator=rfe_model, n_features_to_select=N_FINAL, step=5, verbose=1)
rfe.fit(X_train_stage1, y_train_encoded)

# 獲取選中的特徵
selected_features = [f for f, selected in zip(top_features_stage1, rfe.support_) if selected]

print(f"\n最終選中 {len(selected_features)} 個特徵")

# 分類統計
original_features = [f for f in selected_features if ' ' not in f]
interaction_features = [f for f in selected_features if ' ' in f]

print(f"  原始特徵: {len(original_features)}")
print(f"  交互特徵: {len(interaction_features)}")

print("\n選中的原始特徵:")
for f in original_features:
    print(f"  - {f}")

print("\n選中的交互特徵:")
for f in interaction_features:
    print(f"  - {f}")

X_train_final = X_train_all[selected_features]
X_test_final = X_test_all[selected_features]

# ============================================================
# 6. 儲存精選特徵資料
# ============================================================
print("\n\n【儲存精選特徵資料】")

train_final = X_train_final.copy()
train_final['ASA_Rating'] = y_train.values

train_final.to_csv('train_selected_features.csv', index=False)
X_test_final.to_csv('test_selected_features.csv', index=False)

# 儲存選中的特徵列表
with open('selected_features.txt', 'w') as f:
    f.write(f"# 方法 J: 精選的 {len(selected_features)} 個特徵\n\n")
    f.write("## 原始特徵:\n")
    for feat in original_features:
        f.write(f"  - {feat}\n")
    f.write("\n## 交互特徵:\n")
    for feat in interaction_features:
        f.write(f"  - {feat}\n")

print(f"已儲存 train_selected_features.csv ({train_final.shape})")
print(f"已儲存 test_selected_features.csv ({X_test_final.shape})")
print(f"已儲存 selected_features.txt")

# ============================================================
# 7. 訓練最終模型
# ============================================================
print("\n\n【訓練最終模型】")
print("=" * 70)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)

# 使用方法 E 的最佳超參數
best_params = {
    'n_estimators': 401,
    'max_depth': 8,
    'learning_rate': 0.0424,
    'num_leaves': 61,
    'min_child_samples': 42,
    'subsample': 0.729,
    'colsample_bytree': 0.945,
    'reg_alpha': 0.035,
    'reg_lambda': 0.0002,
    'random_state': 42,
    'class_weight': 'balanced',
    'verbose': -1,
    'n_jobs': -1
}

print("使用超參數 (方法E最佳):")
for key, value in list(best_params.items())[:5]:
    print(f"  {key}: {value}")

model = lgb.LGBMClassifier(**best_params)
model.fit(X_tr, y_tr)

y_pred = model.predict(X_val)
f1 = f1_score(y_val, y_pred, average='macro')
acc = accuracy_score(y_val, y_pred)

print(f"\n驗證集結果:")
print(f"  F1 Macro: {f1:.4f}")
print(f"  Accuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=['ASA 1', 'ASA 2', 'ASA 3', 'ASA 4']))

# ============================================================
# 8. 特徵重要性 (精選後)
# ============================================================
print("\n【精選後特徵重要性 Top 20】")
print("=" * 70)

final_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for i, (_, row) in enumerate(final_importance.head(20).iterrows()):
    is_interaction = "★" if ' ' in row['Feature'] else " "
    print(f"  {i+1:2}. {is_interaction} {row['Feature']:<40} {row['Importance']:.0f}")

# ============================================================
# 9. 5-Fold 交叉驗證
# ============================================================
print("\n\n【5-Fold 交叉驗證】")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    lgb.LGBMClassifier(**best_params),
    X_train_final, y_train_encoded, cv=cv, scoring='f1_macro'
)

print(f"F1 Macro scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================
# 10. 測試集預測
# ============================================================
print("\n\n【測試集預測】")
print("=" * 70)

final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X_train_final, y_train_encoded)

test_predictions = final_model.predict(X_test_final)
test_predictions_original = test_predictions + 1

submission = pd.DataFrame({
    'Id': range(len(test_predictions)),
    'ASA_Rating': test_predictions_original
})

submission.to_csv('submission_selected_features.csv', index=False)
print(f"預測結果已儲存至 submission_selected_features.csv")

print("\n測試集預測分布:")
pred_dist = pd.Series(test_predictions_original).value_counts().sort_index()
for asa, count in pred_dist.items():
    pct = count / len(test_predictions_original) * 100
    print(f"  ASA {asa}: {count} ({pct:.1f}%)")

# ============================================================
# 11. 方法比較
# ============================================================
print("\n\n【方法比較】")
print("=" * 70)

print(f"{'方法':<35} {'特徵數':>8} {'Kaggle Score':>15} {'本地 F1':>12}")
print("-" * 70)
print(f"{'A: Baseline':<35} {'13':>8} {'0.46513':>15} {'0.4517':>12}")
print(f"{'B: 完整特徵':<35} {'40':>8} {'0.52588':>15} {'0.5122':>12}")
print(f"{'C: 交互特徵':<35} {'70':>8} {'0.53120':>15} {'0.4974':>12}")
print(f"{'E: 超參數調優':<35} {'70':>8} {'0.53809':>15} {'0.5002':>12}")
print(f"{'G: 三階交互':<35} {'90':>8} {'0.53598':>15} {'0.4973':>12}")
print(f"{'H: 縱向特徵':<35} {'137':>8} {'0.51656':>15} {'0.4973':>12}")
print(f"{'I: 推測病史':<35} {'86':>8} {'0.53341':>15} {'0.5012':>12}")
print(f"{'J: 特徵精選':<35} {len(selected_features):>8} {'?':>15} {f1:>12.4f}")

print("\n" + "=" * 70)
print(f"方法 J: 特徵精選完成 (從 {len(X_train_all.columns)} 精選到 {len(selected_features)} 個)")
print("=" * 70)
print("\n請將 submission_selected_features.csv 提交到 Kaggle 查看實際分數！")
