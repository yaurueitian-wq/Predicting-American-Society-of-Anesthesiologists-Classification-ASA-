"""
方法 I: 從 Lab_Values 推測過去病史
基於臨床診斷標準，從檢驗值推測可能的共病症

臨床診斷標準參考:
- 慢性腎臟病 (CKD): Creatinine > 1.2 mg/dL (持續)
- 糖尿病: Glucose > 126 mg/dL (空腹) 或 HbA1c > 6.5%
- 貧血: Hemoglobin < 13 g/dL (男) 或 < 12 g/dL (女)
- 肝功能異常: ALT > 40 U/L 或 AST > 40 U/L
- 凝血功能異常: INR > 1.2 或 Platelets < 150 K/uL
- 電解質失衡: Sodium < 136 或 > 145, Potassium < 3.5 或 > 5.0
- 心臟疾病標記: Troponin I 升高
- 感染/發炎: WBC > 11 K/uL 或 CRP 升高
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 載入資料
# ============================================================
print("=" * 70)
print("方法 I: 從 Lab_Values 推測過去病史")
print("=" * 70)

train_df = pd.read_csv('../資料清洗/train_cleaned.csv')
test_df = pd.read_csv('../資料清洗/test_cleaned.csv')

print(f"\n訓練集: {train_df.shape}")
print(f"測試集: {test_df.shape}")

# ============================================================
# 2. 定義檢驗值提取函數
# ============================================================

def extract_lab_value(lab_str, test_name):
    """
    從 Lab_Values 字串提取特定檢驗的數值和狀態

    Args:
        lab_str: Lab_Values 字串
        test_name: 檢驗項目名稱

    Returns:
        dict: {'value': 數值, 'status': 狀態, 'count': 重複次數}
    """
    if pd.isna(lab_str):
        return {'value': None, 'status': None, 'count': 0}

    lab_str = str(lab_str)

    # 嘗試匹配: "項目名稱: 數值 單位 (狀態) [n=次數]"
    # 例如: "Creatinine: 1.4 mg/dL (H) [n=4]"
    pattern = rf'{re.escape(test_name)}:\s*([\d.]+)\s*[^(]*\(([HLNHLL]+)\)(?:\s*\[n=(\d+)\])?'
    match = re.search(pattern, lab_str, re.IGNORECASE)

    if match:
        value = float(match.group(1))
        status = match.group(2)
        count = int(match.group(3)) if match.group(3) else 1
        return {'value': value, 'status': status, 'count': count}

    # 嘗試更簡單的匹配 (無狀態)
    pattern_simple = rf'{re.escape(test_name)}:\s*([\d.]+)'
    match_simple = re.search(pattern_simple, lab_str, re.IGNORECASE)

    if match_simple:
        return {'value': float(match_simple.group(1)), 'status': 'N', 'count': 1}

    return {'value': None, 'status': None, 'count': 0}


def check_abnormal_status(lab_str, test_name, target_status='H'):
    """
    檢查特定檢驗是否有異常狀態

    Args:
        lab_str: Lab_Values 字串
        test_name: 檢驗項目名稱
        target_status: 目標狀態 ('H', 'L', 'HH', 'LL')

    Returns:
        int: 1=有異常, 0=無異常或無資料
    """
    result = extract_lab_value(lab_str, test_name)
    if result['status'] and target_status in result['status']:
        return 1
    return 0

# ============================================================
# 3. 推測病史邏輯
# ============================================================
print("\n【定義臨床診斷標準】")
print("-" * 70)

# 臨床診斷標準 (基於常用參考值)
CLINICAL_CRITERIA = {
    'CKD': {
        'description': '慢性腎臟病',
        'criteria': 'Creatinine > 1.2 mg/dL 或 Urea nitrogen > 20 mg/dL',
        'tests': ['Creatinine', 'Urea nitrogen']
    },
    'Diabetes': {
        'description': '糖尿病',
        'criteria': 'Glucose > 126 mg/dL 或 HbA1c > 6.5%',
        'tests': ['Glucose', 'Hemoglobin A1c/Hemoglobin.total']
    },
    'Anemia': {
        'description': '貧血',
        'criteria': 'Hemoglobin < 12 g/dL 或 Hematocrit < 36%',
        'tests': ['Hemoglobin', 'Hematocrit']
    },
    'Liver_Disease': {
        'description': '肝功能異常',
        'criteria': 'ALT > 40 U/L 或 AST > 40 U/L 或 Bilirubin > 1.2 mg/dL',
        'tests': ['Alanine aminotransferase', 'Aspartate aminotransferase', 'Bilirubin']
    },
    'Coagulopathy': {
        'description': '凝血功能異常',
        'criteria': 'INR > 1.2 或 Platelets < 150 K/uL',
        'tests': ['Coagulation tissue factor induced.INR', 'Platelets']
    },
    'Electrolyte_Imbalance': {
        'description': '電解質失衡',
        'criteria': 'Sodium 異常 或 Potassium 異常',
        'tests': ['Sodium', 'Potassium']
    },
    'Cardiac_Risk': {
        'description': '心臟疾病風險',
        'criteria': 'Troponin I 升高',
        'tests': ['Troponin I.cardiac']
    },
    'Inflammation': {
        'description': '發炎/感染',
        'criteria': 'WBC > 11 K/uL 或 CRP 升高',
        'tests': ['Leukocytes^^corrected for nucleated erythrocytes', 'C reactive protein']
    }
}

for code, info in CLINICAL_CRITERIA.items():
    print(f"  {code}: {info['description']}")
    print(f"      標準: {info['criteria']}")

# ============================================================
# 4. 實作推測函數
# ============================================================
print("\n\n【推測過去病史】")
print("-" * 70)

def infer_medical_history(row):
    """
    從 Lab_Values 和其他資訊推測過去病史

    Returns:
        dict: 各項病史的推測結果 (0/1)
    """
    lab_str = row.get('Lab_Values', '')
    gender = row.get('Gender', 1)  # 1=Female, 2=Male

    history = {
        'hx_ckd': 0,                # 慢性腎臟病
        'hx_diabetes': 0,           # 糖尿病
        'hx_anemia': 0,             # 貧血
        'hx_liver': 0,              # 肝功能異常
        'hx_coagulopathy': 0,       # 凝血異常
        'hx_electrolyte': 0,        # 電解質失衡
        'hx_cardiac': 0,            # 心臟風險
        'hx_inflammation': 0,       # 發炎/感染
        'hx_count': 0,              # 共病數量
        'hx_severity': 0,           # 嚴重程度分數
    }

    if pd.isna(lab_str):
        return history

    lab_str = str(lab_str)

    # 1. 慢性腎臟病 (CKD)
    # Creatinine > 1.2 (男) 或 > 1.0 (女), 或標記為 H/HH
    creatinine = extract_lab_value(lab_str, 'Creatinine')
    bun = extract_lab_value(lab_str, 'Urea nitrogen')

    if creatinine['value']:
        threshold = 1.2 if gender == 2 else 1.0
        if creatinine['value'] > threshold or creatinine['status'] in ['H', 'HH']:
            history['hx_ckd'] = 1
            if creatinine['status'] == 'HH':
                history['hx_severity'] += 2
            else:
                history['hx_severity'] += 1

    if bun['status'] in ['H', 'HH']:
        history['hx_ckd'] = 1
        history['hx_severity'] += 1

    # 2. 糖尿病
    # Glucose > 126 mg/dL 或 HbA1c > 6.5%
    glucose = extract_lab_value(lab_str, 'Glucose')
    hba1c = extract_lab_value(lab_str, 'Hemoglobin A1c/Hemoglobin.total')

    if glucose['value'] and glucose['value'] > 126:
        history['hx_diabetes'] = 1
        history['hx_severity'] += 1
    elif glucose['status'] in ['H', 'HH']:
        history['hx_diabetes'] = 1
        if glucose['status'] == 'HH':
            history['hx_severity'] += 2
        else:
            history['hx_severity'] += 1

    if hba1c['value'] and hba1c['value'] > 6.5:
        history['hx_diabetes'] = 1
        history['hx_severity'] += 2  # HbA1c 更能反映長期血糖控制

    # 3. 貧血
    # Hemoglobin < 13 (男) 或 < 12 (女)
    hgb = extract_lab_value(lab_str, 'Hemoglobin')
    hct = extract_lab_value(lab_str, 'Hematocrit')

    if hgb['value']:
        threshold = 13 if gender == 2 else 12
        if hgb['value'] < threshold or hgb['status'] in ['L', 'LL']:
            history['hx_anemia'] = 1
            if hgb['status'] == 'LL':
                history['hx_severity'] += 2
            else:
                history['hx_severity'] += 1

    if hct['status'] in ['L', 'LL']:
        history['hx_anemia'] = 1
        history['hx_severity'] += 1

    # 4. 肝功能異常
    alt = extract_lab_value(lab_str, 'Alanine aminotransferase')
    ast = extract_lab_value(lab_str, 'Aspartate aminotransferase')
    bili = extract_lab_value(lab_str, 'Bilirubin')

    if alt['status'] in ['H', 'HH'] or ast['status'] in ['H', 'HH']:
        history['hx_liver'] = 1
        history['hx_severity'] += 1

    if bili['status'] in ['H', 'HH']:
        history['hx_liver'] = 1
        history['hx_severity'] += 1

    # 5. 凝血功能異常
    inr = extract_lab_value(lab_str, 'Coagulation tissue factor induced.INR')
    platelets = extract_lab_value(lab_str, 'Platelets')

    if inr['status'] in ['H', 'HH']:
        history['hx_coagulopathy'] = 1
        history['hx_severity'] += 1

    if platelets['status'] in ['L', 'LL']:
        history['hx_coagulopathy'] = 1
        if platelets['status'] == 'LL':
            history['hx_severity'] += 2
        else:
            history['hx_severity'] += 1

    # 6. 電解質失衡
    sodium = extract_lab_value(lab_str, 'Sodium')
    potassium = extract_lab_value(lab_str, 'Potassium')

    if sodium['status'] in ['H', 'HH', 'L', 'LL']:
        history['hx_electrolyte'] = 1
        history['hx_severity'] += 1

    if potassium['status'] in ['H', 'HH', 'L', 'LL']:
        history['hx_electrolyte'] = 1
        if potassium['status'] in ['HH', 'LL']:
            history['hx_severity'] += 2  # 鉀離子異常更危險
        else:
            history['hx_severity'] += 1

    # 7. 心臟風險
    troponin = extract_lab_value(lab_str, 'Troponin I.cardiac')

    if troponin['status'] in ['H', 'HH']:
        history['hx_cardiac'] = 1
        history['hx_severity'] += 3  # 心臟風險權重較高

    # 8. 發炎/感染
    wbc = extract_lab_value(lab_str, 'Leukocytes^^corrected for nucleated erythrocytes')
    crp = extract_lab_value(lab_str, 'C reactive protein')

    if wbc['status'] in ['H', 'HH']:
        history['hx_inflammation'] = 1
        history['hx_severity'] += 1

    if crp['status'] in ['H', 'HH']:
        history['hx_inflammation'] = 1
        history['hx_severity'] += 1

    # 計算共病數量
    history['hx_count'] = sum([
        history['hx_ckd'], history['hx_diabetes'], history['hx_anemia'],
        history['hx_liver'], history['hx_coagulopathy'], history['hx_electrolyte'],
        history['hx_cardiac'], history['hx_inflammation']
    ])

    return history

# 應用推測函數
print("  提取訓練集病史...")
train_history = train_df.apply(infer_medical_history, axis=1).apply(pd.Series)
print("  提取測試集病史...")
test_history = test_df.apply(infer_medical_history, axis=1).apply(pd.Series)

print(f"\n  病史特徵數: {len(train_history.columns)}")
print(f"  特徵列表: {list(train_history.columns)}")

# ============================================================
# 5. 分析推測結果與 ASA 的關聯
# ============================================================
print("\n\n【病史推測結果分析】")
print("=" * 70)

train_history['ASA_Rating'] = train_df['ASA_Rating']

print("\n各病史在不同 ASA 分級的盛行率:")
print("-" * 70)
print(f"{'病史':<20} {'ASA 1':>8} {'ASA 2':>8} {'ASA 3':>8} {'ASA 4':>8} {'整體':>8}")
print("-" * 70)

history_features = ['hx_ckd', 'hx_diabetes', 'hx_anemia', 'hx_liver',
                   'hx_coagulopathy', 'hx_electrolyte', 'hx_cardiac', 'hx_inflammation']

for hx in history_features:
    rates = []
    for asa in [1, 2, 3, 4]:
        subset = train_history[train_history['ASA_Rating'] == asa]
        rate = subset[hx].mean() * 100
        rates.append(f"{rate:6.1f}%")
    overall = train_history[hx].mean() * 100

    name_map = {
        'hx_ckd': '慢性腎臟病',
        'hx_diabetes': '糖尿病',
        'hx_anemia': '貧血',
        'hx_liver': '肝功能異常',
        'hx_coagulopathy': '凝血異常',
        'hx_electrolyte': '電解質失衡',
        'hx_cardiac': '心臟風險',
        'hx_inflammation': '發炎/感染'
    }

    print(f"{name_map.get(hx, hx):<18} {rates[0]:>8} {rates[1]:>8} {rates[2]:>8} {rates[3]:>8} {overall:>6.1f}%")

# 共病數量分析
print("\n\n共病數量與 ASA 的關係:")
print("-" * 70)
for asa in [1, 2, 3, 4]:
    subset = train_history[train_history['ASA_Rating'] == asa]
    avg_count = subset['hx_count'].mean()
    avg_severity = subset['hx_severity'].mean()
    print(f"  ASA {asa}: 平均共病數 = {avg_count:.2f}, 平均嚴重度 = {avg_severity:.2f}")

# 移除暫時加入的 ASA_Rating
train_history = train_history.drop('ASA_Rating', axis=1)

# ============================================================
# 6. 載入方法B特徵並合併
# ============================================================
print("\n\n【載入方法B特徵並合併病史特徵】")
print("-" * 70)

train_b = pd.read_csv('../方法B_完整特徵/train_features.csv')
test_b = pd.read_csv('../方法B_完整特徵/test_features.csv')

target = 'ASA_Rating'
y_train = train_b[target].copy()

feature_cols_b = [col for col in train_b.columns if col != target]
X_train_b = train_b[feature_cols_b].copy()
X_test_b = test_b[feature_cols_b].copy()

# 合併病史特徵
X_train_combined = pd.concat([X_train_b.reset_index(drop=True),
                              train_history.reset_index(drop=True)], axis=1)
X_test_combined = pd.concat([X_test_b.reset_index(drop=True),
                             test_history.reset_index(drop=True)], axis=1)

print(f"  方法B特徵數: {len(feature_cols_b)}")
print(f"  新增病史特徵: {len(train_history.columns)}")
print(f"  合併後特徵數: {len(X_train_combined.columns)}")

# ============================================================
# 7. 資料前處理
# ============================================================
print("\n【資料前處理】")

categorical_features = ['Gender', 'ICU_Patient', 'Anesthesia_Method', 'Patient_Source',
                        'Surgery_Category', 'Drug_Category', 'Route_Standardized',
                        'Surgery_Procedure_Type']
numeric_features = [col for col in X_train_combined.columns if col not in categorical_features]

# 處理缺失值
for col in categorical_features:
    if col in X_train_combined.columns:
        X_train_combined[col] = X_train_combined[col].fillna('Unknown')
        X_test_combined[col] = X_test_combined[col].fillna('Unknown')

for col in numeric_features:
    if col in X_train_combined.columns:
        X_train_combined[col] = X_train_combined[col].fillna(0)
        X_test_combined[col] = X_test_combined[col].fillna(0)

# Label Encoding
label_encoders = {}
for col in categorical_features:
    if col in X_train_combined.columns:
        le = LabelEncoder()
        all_values = pd.concat([X_train_combined[col], X_test_combined[col]]).unique()
        le.fit(all_values)
        X_train_combined[col] = le.transform(X_train_combined[col])
        X_test_combined[col] = le.transform(X_test_combined[col])
        label_encoders[col] = le

# 標準化數值特徵
scaler = StandardScaler()
num_cols_to_scale = [col for col in numeric_features if col in X_train_combined.columns]
X_train_combined[num_cols_to_scale] = scaler.fit_transform(X_train_combined[num_cols_to_scale])
X_test_combined[num_cols_to_scale] = scaler.transform(X_test_combined[num_cols_to_scale])

# 0-based 標籤
y_train_encoded = y_train - 1

print(f"  前處理後訓練集: {X_train_combined.shape}")
print(f"  前處理後測試集: {X_test_combined.shape}")

# ============================================================
# 8. 加入交互特徵 (與方法C相同)
# ============================================================
print("\n【加入交互特徵】")
print("-" * 70)

important_numeric = [
    'Age', 'BMI', 'Surgery_Count',
    'lab_abnormal_total', 'lab_count',
    'med_count', 'catheter_count',
    'hx_count', 'hx_severity'  # 新增病史相關交互
]
important_numeric = [f for f in important_numeric if f in X_train_combined.columns]

print(f"用於交互的特徵: {important_numeric}")

X_train_interact = X_train_combined[important_numeric].copy()
X_test_interact = X_test_combined[important_numeric].copy()

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_interact)
X_test_poly = poly.transform(X_test_interact)

poly_feature_names = poly.get_feature_names_out(important_numeric)
interaction_cols = [col for col in poly_feature_names if ' ' in col]

print(f"交互特徵數: {len(interaction_cols)}")

X_train_poly_df = pd.DataFrame(X_train_poly, columns=poly_feature_names)
X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly_feature_names)

X_train_final = pd.concat([
    X_train_combined.reset_index(drop=True),
    X_train_poly_df[interaction_cols].reset_index(drop=True)
], axis=1)

X_test_final = pd.concat([
    X_test_combined.reset_index(drop=True),
    X_test_poly_df[interaction_cols].reset_index(drop=True)
], axis=1)

print(f"最終特徵數: {X_test_final.shape[1]}")

# ============================================================
# 9. 儲存特徵資料
# ============================================================
print("\n【儲存特徵資料】")

train_final = X_train_final.copy()
train_final['ASA_Rating'] = y_train.values

train_final.to_csv('train_inferred_history.csv', index=False)
X_test_final.to_csv('test_inferred_history.csv', index=False)

print(f"  已儲存 train_inferred_history.csv ({train_final.shape})")
print(f"  已儲存 test_inferred_history.csv ({X_test_final.shape})")

# ============================================================
# 10. 訓練模型
# ============================================================
print("\n\n【訓練模型】")
print("=" * 70)

# 切分驗證集
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)

# 使用方法E的最佳超參數
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

# 驗證集評估
y_pred = model.predict(X_val)
f1 = f1_score(y_val, y_pred, average='macro')
acc = accuracy_score(y_val, y_pred)

print(f"\n驗證集結果:")
print(f"  F1 Macro: {f1:.4f}")
print(f"  Accuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=['ASA 1', 'ASA 2', 'ASA 3', 'ASA 4']))

# ============================================================
# 11. 特徵重要性分析
# ============================================================
print("\n【特徵重要性 Top 25】")
print("=" * 70)

importance_df = pd.DataFrame({
    'Feature': X_train_final.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

# 顯示病史特徵的重要性排名
print("\n病史特徵重要性排名:")
hx_features = [f for f in importance_df['Feature'] if f.startswith('hx_')]
for i, feat in enumerate(hx_features):
    rank = importance_df[importance_df['Feature'] == feat].index[0] + 1
    imp = importance_df[importance_df['Feature'] == feat]['Importance'].values[0]
    print(f"  {feat:<20} 排名: {rank:>3}, 重要性: {imp:.4f}")

print("\n整體 Top 15 特徵:")
for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
    is_hx = "★" if row['Feature'].startswith('hx_') else " "
    print(f"  {i+1:2}. {is_hx} {row['Feature']:<35} {row['Importance']:.4f}")

# ============================================================
# 12. 5-Fold 交叉驗證
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
# 13. 測試集預測
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

submission.to_csv('submission_inferred_history.csv', index=False)
print(f"預測結果已儲存至 submission_inferred_history.csv")

print("\n測試集預測分布:")
pred_dist = pd.Series(test_predictions_original).value_counts().sort_index()
for asa, count in pred_dist.items():
    pct = count / len(test_predictions_original) * 100
    print(f"  ASA {asa}: {count} ({pct:.1f}%)")

# ============================================================
# 14. 方法比較
# ============================================================
print("\n\n【方法比較】")
print("=" * 70)

print(f"{'方法':<35} {'特徵數':>8} {'Kaggle Score':>15} {'本地 F1':>12}")
print("-" * 70)
print(f"{'A: Baseline':<35} {'13':>8} {'0.46513':>15} {'0.4517':>12}")
print(f"{'B: 完整特徵':<35} {'40':>8} {'0.52588':>15} {'0.5122':>12}")
print(f"{'C: 交互特徵 (degree=2)':<35} {'70':>8} {'0.53120':>15} {'0.4974':>12}")
print(f"{'E: 超參數調優':<35} {'70':>8} {'0.53809':>15} {'0.5002':>12}")
print(f"{'H: 縱向特徵':<35} {'137':>8} {'0.51656':>15} {'0.4973':>12}")
print(f"{'I: 推測病史':<35} {X_train_final.shape[1]:>8} {'?':>15} {f1:>12.4f}")

print("\n" + "=" * 70)
print("方法 I: 推測病史完成")
print("=" * 70)
print("\n請將 submission_inferred_history.csv 提交到 Kaggle 查看實際分數！")
