"""
方法 B: 完整特徵工程
從文字欄位提取更多特徵，建立更完整的特徵集

新增特徵:
1. Lab_Values: 異常指標數量、特定檢驗值
2. Medication_Usage: 用藥數量、慢性病用藥標記
3. Catheter_Use: 導管數量、導管類型
4. 其他衍生特徵
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 載入資料
# ============================================================
print("=" * 70)
print("方法 B: 完整特徵工程")
print("=" * 70)

train_df = pd.read_csv('../資料清洗/train_cleaned.csv')
test_df = pd.read_csv('../資料清洗/test_cleaned.csv')

print(f"\n訓練集: {train_df.shape}")
print(f"測試集: {test_df.shape}")

# ============================================================
# 2. Lab_Values 特徵提取
# ============================================================
print("\n【Lab_Values 特徵提取】")
print("-" * 70)

def extract_lab_features(lab_str):
    """從 Lab_Values 提取特徵"""
    features = {
        'lab_count': 0,           # 檢驗項目數量
        'lab_abnormal_H': 0,      # 高於正常值數量
        'lab_abnormal_L': 0,      # 低於正常值數量
        'lab_critical_HH': 0,     # 危急高值數量
        'lab_critical_LL': 0,     # 危急低值數量
        'lab_has_data': 0,        # 是否有檢驗資料
    }

    if pd.isna(lab_str):
        return features

    features['lab_has_data'] = 1

    # 計算各狀態數量
    # 格式: "項目: 數值 單位 (狀態) [n=重複次數]"
    pattern = r'\(([HLN]+)\)'
    matches = re.findall(pattern, str(lab_str))

    features['lab_count'] = len(matches)

    for status in matches:
        if status == 'HH':
            features['lab_critical_HH'] += 1
        elif status == 'LL':
            features['lab_critical_LL'] += 1
        elif status == 'H':
            features['lab_abnormal_H'] += 1
        elif status == 'L':
            features['lab_abnormal_L'] += 1

    return features

# 提取 Lab 特徵
print("  提取 Lab_Values 特徵...")
train_lab_features = train_df['Lab_Values'].apply(extract_lab_features).apply(pd.Series)
test_lab_features = test_df['Lab_Values'].apply(extract_lab_features).apply(pd.Series)

# 計算總異常數
train_lab_features['lab_abnormal_total'] = (
    train_lab_features['lab_abnormal_H'] +
    train_lab_features['lab_abnormal_L'] +
    train_lab_features['lab_critical_HH'] +
    train_lab_features['lab_critical_LL']
)
test_lab_features['lab_abnormal_total'] = (
    test_lab_features['lab_abnormal_H'] +
    test_lab_features['lab_abnormal_L'] +
    test_lab_features['lab_critical_HH'] +
    test_lab_features['lab_critical_LL']
)

print(f"  Lab 特徵數: {len(train_lab_features.columns)}")
print(f"  特徵: {list(train_lab_features.columns)}")

# ============================================================
# 3. Medication_Usage 特徵提取
# ============================================================
print("\n【Medication_Usage 特徵提取】")
print("-" * 70)

def extract_med_features(row):
    """從 Medication_Usage 和相關欄位提取特徵"""
    features = {
        'med_count': 0,           # 用藥數量
        'has_chronic_med': 0,     # 是否有慢性病用藥
        'has_cardiac_med': 0,     # 是否有心臟用藥
        'has_diabetes_med': 0,    # 是否有糖尿病用藥
        'has_anticoagulant': 0,   # 是否有抗凝血劑
        'has_opioid': 0,          # 是否有鴉片類藥物
        'has_sedative': 0,        # 是否有鎮靜劑
    }

    med_str = str(row.get('Medication_Usage', ''))
    drug_cat = str(row.get('Drug_Category', ''))
    drug_name = str(row.get('Drug_Standardized', '')).lower()

    # 用藥數量 (以逗號分隔)
    if pd.notna(row.get('Medication_Usage')) and med_str != 'nan':
        features['med_count'] = len(med_str.split(','))

    # 慢性病用藥標記
    chronic_categories = ['ANTIHYPERTENSIVES', 'CARDIAC', 'DIURETICS', 'ANTIARRHYTHMICS']
    if drug_cat in chronic_categories:
        features['has_chronic_med'] = 1
        features['has_cardiac_med'] = 1

    # 糖尿病用藥
    if drug_cat == 'INSULIN' or 'metformin' in drug_name or 'glipizide' in drug_name:
        features['has_diabetes_med'] = 1
        features['has_chronic_med'] = 1

    # 抗凝血劑
    if drug_cat == 'ANTICOAGULANTS':
        features['has_anticoagulant'] = 1

    # 鴉片類藥物
    if drug_cat == 'OPIOID_ANALGESICS':
        features['has_opioid'] = 1

    # 鎮靜劑
    if drug_cat == 'SEDATIVES_HYPNOTICS':
        features['has_sedative'] = 1

    return features

print("  提取 Medication 特徵...")
train_med_features = train_df.apply(extract_med_features, axis=1).apply(pd.Series)
test_med_features = test_df.apply(extract_med_features, axis=1).apply(pd.Series)

print(f"  Medication 特徵數: {len(train_med_features.columns)}")
print(f"  特徵: {list(train_med_features.columns)}")

# ============================================================
# 4. Catheter_Use 特徵提取
# ============================================================
print("\n【Catheter_Use 特徵提取】")
print("-" * 70)

def extract_catheter_features(cath_str):
    """從 Catheter_Use 提取特徵"""
    features = {
        'catheter_count': 0,      # 導管總數
        'has_catheter': 0,        # 是否有導管
        'has_piv': 0,             # 是否有周邊靜脈導管
        'has_urinary': 0,         # 是否有導尿管
        'has_cvc': 0,             # 是否有中心靜脈導管
        'has_arterial': 0,        # 是否有動脈管路
        'has_chest_tube': 0,      # 是否有胸管
        'has_wound': 0,           # 是否有傷口相關
    }

    if pd.isna(cath_str):
        return features

    cath_str = str(cath_str).upper()
    features['has_catheter'] = 1

    # 計算導管數量 (以逗號分隔)
    catheters = cath_str.split(',')
    features['catheter_count'] = len(catheters)

    # 各類型導管
    if 'PIV' in cath_str or 'PERIPHERAL IV' in cath_str:
        features['has_piv'] = 1

    if 'URINARY' in cath_str:
        features['has_urinary'] = 1

    if 'CVC' in cath_str or 'CENTRAL' in cath_str or 'PICC' in cath_str:
        features['has_cvc'] = 1

    if 'ARTERIAL' in cath_str or 'ART' in cath_str:
        features['has_arterial'] = 1

    if 'CHEST TUBE' in cath_str:
        features['has_chest_tube'] = 1

    if 'WOUND' in cath_str or 'INCISION' in cath_str or 'DRAIN' in cath_str:
        features['has_wound'] = 1

    return features

print("  提取 Catheter 特徵...")
train_cath_features = train_df['Catheter_Use'].apply(extract_catheter_features).apply(pd.Series)
test_cath_features = test_df['Catheter_Use'].apply(extract_catheter_features).apply(pd.Series)

print(f"  Catheter 特徵數: {len(train_cath_features.columns)}")
print(f"  特徵: {list(train_cath_features.columns)}")

# ============================================================
# 5. 其他衍生特徵
# ============================================================
print("\n【其他衍生特徵】")
print("-" * 70)

def extract_other_features(row):
    """提取其他衍生特徵"""
    features = {
        'age_group': 0,           # 年齡分組
        'bmi_category': 0,        # BMI 分類
        'is_elderly': 0,          # 是否為老年人 (>=65)
        'is_obese': 0,            # 是否肥胖 (BMI >= 30)
        'is_morbid_obese': 0,     # 是否病態肥胖 (BMI >= 40)
    }

    # 年齡分組: 0=17-40, 1=40-60, 2=60-75, 3=75+
    age = row.get('Age', 0)
    if age < 40:
        features['age_group'] = 0
    elif age < 60:
        features['age_group'] = 1
    elif age < 75:
        features['age_group'] = 2
    else:
        features['age_group'] = 3

    features['is_elderly'] = 1 if age >= 65 else 0

    # BMI 分類: 0=<30, 1=30-35, 2=35-40, 3=>=40
    bmi = row.get('BMI', 0)
    if bmi < 30:
        features['bmi_category'] = 0
    elif bmi < 35:
        features['bmi_category'] = 1
    elif bmi < 40:
        features['bmi_category'] = 2
    else:
        features['bmi_category'] = 3

    features['is_obese'] = 1 if bmi >= 30 else 0
    features['is_morbid_obese'] = 1 if bmi >= 40 else 0

    return features

print("  提取其他衍生特徵...")
train_other_features = train_df.apply(extract_other_features, axis=1).apply(pd.Series)
test_other_features = test_df.apply(extract_other_features, axis=1).apply(pd.Series)

print(f"  其他特徵數: {len(train_other_features.columns)}")
print(f"  特徵: {list(train_other_features.columns)}")

# ============================================================
# 6. 合併所有特徵
# ============================================================
print("\n【合併所有特徵】")
print("-" * 70)

# 原始特徵
numeric_features = ['Age', 'HEIGHT', 'WEIGHT', 'BMI', 'Surgery_Count']
categorical_features = ['Gender', 'ICU_Patient', 'Anesthesia_Method', 'Patient_Source',
                        'Surgery_Category', 'Drug_Category', 'Route_Standardized',
                        'Surgery_Procedure_Type']

# 合併訓練集
train_combined = pd.concat([
    train_df[numeric_features + categorical_features],
    train_lab_features,
    train_med_features,
    train_cath_features,
    train_other_features
], axis=1)

# 合併測試集
test_combined = pd.concat([
    test_df[numeric_features + categorical_features],
    test_lab_features,
    test_med_features,
    test_cath_features,
    test_other_features
], axis=1)

# 目標變數
y_train = train_df['ASA_Rating']

print(f"  原始特徵: {len(numeric_features + categorical_features)}")
print(f"  Lab 特徵: {len(train_lab_features.columns)}")
print(f"  Medication 特徵: {len(train_med_features.columns)}")
print(f"  Catheter 特徵: {len(train_cath_features.columns)}")
print(f"  其他特徵: {len(train_other_features.columns)}")
print(f"  總特徵數: {len(train_combined.columns)}")

print(f"\n  訓練集 shape: {train_combined.shape}")
print(f"  測試集 shape: {test_combined.shape}")

# ============================================================
# 7. 儲存特徵工程後的資料
# ============================================================
print("\n【儲存特徵工程後的資料】")
print("-" * 70)

# 加入目標變數
train_combined['ASA_Rating'] = y_train

# 儲存
train_combined.to_csv('train_features.csv', index=False)
test_combined.to_csv('test_features.csv', index=False)

print(f"  已儲存 train_features.csv ({train_combined.shape})")
print(f"  已儲存 test_features.csv ({test_combined.shape})")

# ============================================================
# 8. 特徵統計
# ============================================================
print("\n\n【特徵統計摘要】")
print("=" * 70)

print("\n新增特徵與 ASA 的關聯:")
new_features = list(train_lab_features.columns) + list(train_med_features.columns) + \
               list(train_cath_features.columns) + list(train_other_features.columns)

for feat in new_features[:10]:  # 顯示前 10 個
    if feat in train_combined.columns:
        corr = train_combined[feat].corr(train_combined['ASA_Rating'])
        print(f"  {feat:<25}: corr = {corr:+.3f}")

print("\n" + "=" * 70)
print("特徵工程完成")
print("=" * 70)
