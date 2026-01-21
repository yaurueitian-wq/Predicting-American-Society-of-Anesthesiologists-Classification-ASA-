"""
方法 H: 縱向特徵工程
從 Lab_Values 的重複測量中提取縱向特徵，推測患者病史

核心發現:
- Lab_Values 包含同一檢驗項目的多次測量 (最多 123 次)
- ASA 4 患者平均重複測量 21.9 次，ASA 1 僅 2.5 次
- 持續性異常與 ASA 分級高度相關

新增特徵 (以 h_ 前綴命名，保留方法B所有欄位):
1. h_max_lab_repeats - 最大重複測量次數
2. h_creat_* - Creatinine 腎功能相關
3. h_hgb_* - Hemoglobin 貧血相關
4. h_glucose_* - Glucose 血糖相關
5. h_sodium_* - Sodium 電解質相關
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 載入資料
# ============================================================
print("=" * 70)
print("方法 H: 縱向特徵工程 (Lab_Values 深度解析)")
print("=" * 70)

# 載入方法B的特徵工程資料 (保留所有舊欄位)
train_b = pd.read_csv('../方法B_完整特徵/train_features.csv')
test_b = pd.read_csv('../方法B_完整特徵/test_features.csv')

# 載入原始清洗後資料 (需要 Lab_Values 原始文字)
train_raw = pd.read_csv('../資料清洗/train_cleaned.csv')
test_raw = pd.read_csv('../資料清洗/test_cleaned.csv')

print(f"\n【資料載入】")
print(f"方法B訓練集: {train_b.shape}")
print(f"方法B測試集: {test_b.shape}")
print(f"原始訓練集: {train_raw.shape}")
print(f"原始測試集: {test_raw.shape}")

target = 'ASA_Rating'
y_train = train_b[target].copy()

# ============================================================
# 2. 定義縱向特徵提取函數
# ============================================================
print("\n【定義縱向特徵提取函數】")
print("-" * 70)

def count_test_repeats(lab_str):
    """計算最大重複測量次數"""
    if pd.isna(lab_str):
        return 0

    entries = str(lab_str).split(',')
    test_names = []

    for entry in entries:
        match = re.match(r'^([^:]+):', entry.strip())
        if match:
            test_names.append(match.group(1).strip())

    if not test_names:
        return 0

    counts = Counter(test_names)
    return max(counts.values())


def extract_longitudinal_features(lab_str, test_name):
    """
    提取特定檢驗項目的縱向特徵
    """
    features = {
        'count': 0,
        'persistently_abnormal': 0,
        'trend': 0,
        'range': 0,
        'abnormal_ratio': 0
    }

    if pd.isna(lab_str):
        return features

    entries = str(lab_str).split(',')
    values = []
    flags = []

    for entry in entries:
        # 匹配格式: "TestName: value unit (flag)"
        pattern = rf'^{re.escape(test_name)}:\s*([\d.]+).*?\(([HLN]+)\)'
        match = re.search(pattern, entry.strip(), re.IGNORECASE)
        if match:
            try:
                values.append(float(match.group(1)))
                flags.append(match.group(2))
            except ValueError:
                continue

    if not values:
        return features

    features['count'] = len(values)
    features['range'] = max(values) - min(values)

    # 異常比率
    abnormal_count = len([f for f in flags if f != 'N'])
    features['abnormal_ratio'] = abnormal_count / len(flags)

    # 持續性異常 (>=80% 異常)
    features['persistently_abnormal'] = 1 if features['abnormal_ratio'] >= 0.8 else 0

    # 變化趨勢
    if len(values) > 1 and values[0] != 0:
        features['trend'] = (values[-1] - values[0]) / values[0]

    return features


# ============================================================
# 3. 提取縱向特徵 (訓練集和測試集)
# ============================================================
print("\n【提取縱向特徵】")
print("-" * 70)

# 要分析的關鍵檢驗項目
key_tests = {
    'Creatinine': 'creat',      # 腎功能
    'Hemoglobin': 'hgb',        # 貧血
    'Glucose': 'glucose',       # 血糖/糖尿病
    'Sodium': 'sodium',         # 電解質
    'Potassium': 'potassium',   # 電解質
    'Platelets': 'plt',         # 凝血功能
}


def extract_all_longitudinal_features(df_raw, dataset_name=""):
    """從原始資料提取所有縱向特徵"""
    print(f"\n處理 {dataset_name}...")

    results = pd.DataFrame(index=df_raw.index)

    # 1. 最大重複測量次數
    print("  計算 h_max_lab_repeats...")
    results['h_max_lab_repeats'] = df_raw['Lab_Values'].apply(count_test_repeats)

    # 2. 各檢驗項目的縱向特徵
    for test_name, prefix in key_tests.items():
        print(f"  提取 {test_name} 縱向特徵...")

        longitudinal_data = df_raw['Lab_Values'].apply(
            lambda x: extract_longitudinal_features(x, test_name)
        ).apply(pd.Series)

        # 重命名欄位 (以 h_ 前綴)
        for col in longitudinal_data.columns:
            results[f'h_{prefix}_{col}'] = longitudinal_data[col]

    print(f"  完成! 縱向特徵數: {len(results.columns)}")
    return results


# 提取訓練集縱向特徵
train_longitudinal = extract_all_longitudinal_features(train_raw, "訓練集")

# 提取測試集縱向特徵
test_longitudinal = extract_all_longitudinal_features(test_raw, "測試集")

# 驗證兩者欄位一致
assert list(train_longitudinal.columns) == list(test_longitudinal.columns), "訓練集和測試集欄位不一致!"
print(f"\n縱向特徵欄位一致性檢查: 通過")

# ============================================================
# 4. 合併方法B特徵與新縱向特徵
# ============================================================
print("\n【合併特徵】")
print("-" * 70)

# 移除目標變數
feature_cols_b = [col for col in train_b.columns if col != target]

# 合併: 方法B特徵 + 縱向特徵
train_combined = pd.concat([
    train_b[feature_cols_b].reset_index(drop=True),
    train_longitudinal.reset_index(drop=True)
], axis=1)

test_combined = pd.concat([
    test_b[feature_cols_b].reset_index(drop=True),
    test_longitudinal.reset_index(drop=True)
], axis=1)

print(f"方法B特徵數: {len(feature_cols_b)}")
print(f"新增縱向特徵數: {len(train_longitudinal.columns)}")
print(f"合併後訓練集: {train_combined.shape}")
print(f"合併後測試集: {test_combined.shape}")

# 顯示新增的縱向特徵
print("\n新增的縱向特徵:")
for col in train_longitudinal.columns:
    print(f"  - {col}")

# ============================================================
# 5. 資料前處理
# ============================================================
print("\n【資料前處理】")
print("-" * 70)

categorical_features = ['Gender', 'ICU_Patient', 'Anesthesia_Method', 'Patient_Source',
                        'Surgery_Category', 'Drug_Category', 'Route_Standardized',
                        'Surgery_Procedure_Type']
numeric_features = [col for col in train_combined.columns if col not in categorical_features]

# 處理缺失值
for col in categorical_features:
    if col in train_combined.columns:
        train_combined[col] = train_combined[col].fillna('Unknown')
        test_combined[col] = test_combined[col].fillna('Unknown')

for col in numeric_features:
    train_combined[col] = train_combined[col].fillna(0)
    test_combined[col] = test_combined[col].fillna(0)

# Label Encoding
label_encoders = {}
for col in categorical_features:
    if col in train_combined.columns:
        le = LabelEncoder()
        all_values = pd.concat([train_combined[col], test_combined[col]]).unique()
        le.fit(all_values)
        train_combined[col] = le.transform(train_combined[col])
        test_combined[col] = le.transform(test_combined[col])
        label_encoders[col] = le

# 標準化數值特徵 (保存 scaler 以確保訓練/測試一致)
scaler = StandardScaler()
train_combined[numeric_features] = scaler.fit_transform(train_combined[numeric_features])
test_combined[numeric_features] = scaler.transform(test_combined[numeric_features])

# 0-based 標籤
y_train_encoded = y_train - 1

print(f"前處理後訓練集: {train_combined.shape}")
print(f"前處理後測試集: {test_combined.shape}")

# ============================================================
# 6. 產生二階交互特徵
# ============================================================
print("\n【產生二階交互特徵】")
print("-" * 70)

# 選擇重要數值特徵進行交互 (包含新增縱向特徵)
important_numeric = [
    'Age', 'BMI', 'Surgery_Count',
    'lab_abnormal_total', 'lab_count',
    'med_count', 'catheter_count',
    'is_elderly', 'is_obese',
    'h_max_lab_repeats',
    'h_creat_persistently_abnormal',
    'h_hgb_persistently_abnormal'
]
important_numeric = [f for f in important_numeric if f in train_combined.columns]

print(f"用於交互的特徵 ({len(important_numeric)} 個):")
for f in important_numeric:
    print(f"  - {f}")

# 提取用於交互的特徵
X_train_interact = train_combined[important_numeric].copy()
X_test_interact = test_combined[important_numeric].copy()

# PolynomialFeatures (degree=2)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_interact)
X_test_poly = poly.transform(X_test_interact)

# 獲取交互特徵名稱
poly_feature_names = poly.get_feature_names_out(important_numeric)

# 只保留交互項 (排除原始特徵)
interaction_cols = [col for col in poly_feature_names if ' ' in col]
interaction_idx = [i for i, col in enumerate(poly_feature_names) if ' ' in col]

X_train_interactions = pd.DataFrame(
    X_train_poly[:, interaction_idx],
    columns=interaction_cols
)
X_test_interactions = pd.DataFrame(
    X_test_poly[:, interaction_idx],
    columns=interaction_cols
)

print(f"二階交互特徵數: {len(interaction_cols)}")

# 合併所有特徵
X_train_final = pd.concat([
    train_combined.reset_index(drop=True),
    X_train_interactions.reset_index(drop=True)
], axis=1)

X_test_final = pd.concat([
    test_combined.reset_index(drop=True),
    X_test_interactions.reset_index(drop=True)
], axis=1)

print(f"\n最終訓練集: {X_train_final.shape}")
print(f"最終測試集: {X_test_final.shape}")

# 驗證欄位一致
assert list(X_train_final.columns) == list(X_test_final.columns), "最終訓練/測試集欄位不一致!"
print("最終欄位一致性檢查: 通過")

# ============================================================
# 7. 儲存特徵工程後的資料
# ============================================================
print("\n【儲存特徵工程後的資料】")
print("-" * 70)

# 訓練集加入目標變數
train_h = X_train_final.copy()
train_h['ASA_Rating'] = y_train.values

train_h.to_csv('train_longitudinal_features.csv', index=False)
X_test_final.to_csv('test_longitudinal_features.csv', index=False)

print(f"已儲存 train_longitudinal_features.csv ({train_h.shape})")
print(f"已儲存 test_longitudinal_features.csv ({X_test_final.shape})")

# 儲存新增特徵列表
with open('new_features_list.txt', 'w') as f:
    f.write("# 方法H: 新增的特徵\n\n")
    f.write("## 縱向特徵:\n")
    for col in train_longitudinal.columns:
        f.write(f"  - {col}\n")
    f.write(f"\n## 交互特徵 ({len(interaction_cols)} 個):\n")
    for col in interaction_cols[:20]:
        f.write(f"  - {col}\n")
    if len(interaction_cols) > 20:
        f.write(f"  ... 等共 {len(interaction_cols)} 個\n")

print("已儲存 new_features_list.txt")

# ============================================================
# 8. 模型訓練與評估
# ============================================================
print("\n【模型訓練與評估】")
print("=" * 70)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)

print(f"訓練集: {X_tr.shape}")
print(f"驗證集: {X_val.shape}")

# 方法E最佳超參數
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

print("\n使用方法E最佳超參數訓練...")
model = lgb.LGBMClassifier(**best_params)
model.fit(X_tr, y_tr)

# 驗證集評估
y_pred = model.predict(X_val)
f1 = f1_score(y_val, y_pred, average='macro')
acc = accuracy_score(y_val, y_pred)
kappa = cohen_kappa_score(y_val, y_pred)

print(f"\n驗證集結果:")
print(f"  F1 Macro: {f1:.4f}")
print(f"  Accuracy: {acc:.4f}")
print(f"  Kappa: {kappa:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=['ASA 1', 'ASA 2', 'ASA 3', 'ASA 4']))

# ============================================================
# 9. 特徵重要性分析
# ============================================================
print("\n【特徵重要性 Top 20】")
print("=" * 70)

importance_df = pd.DataFrame({
    'Feature': X_train_final.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n整體 Top 15:")
for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
    is_new = 'h_' in row['Feature'] or ' ' in row['Feature']
    marker = "[NEW]" if is_new else ""
    print(f"  {i+1:2}. {row['Feature']:<40} {row['Importance']:.0f} {marker}")

print("\n縱向特徵 Top 10:")
h_features = importance_df[importance_df['Feature'].str.startswith('h_')]
for i, (_, row) in enumerate(h_features.head(10).iterrows()):
    print(f"  {i+1:2}. {row['Feature']:<40} {row['Importance']:.0f}")

# ============================================================
# 10. 5-Fold 交叉驗證
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
# 11. 測試集預測
# ============================================================
print("\n\n【測試集預測】")
print("=" * 70)

# 使用完整訓練集訓練
print("使用完整訓練集重新訓練...")
final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X_train_final, y_train_encoded)

# 預測測試集
test_predictions = final_model.predict(X_test_final)
test_predictions_original = test_predictions + 1  # 0-3 -> 1-4

# Kaggle 格式輸出
submission = pd.DataFrame({
    'Id': range(len(test_predictions)),
    'ASA_Rating': test_predictions_original
})

submission.to_csv('submission_longitudinal.csv', index=False)
print(f"已儲存 submission_longitudinal.csv ({len(submission)} 筆)")

# 預測分布
print("\n測試集預測分布:")
pred_dist = pd.Series(test_predictions_original).value_counts().sort_index()
for asa, count in pred_dist.items():
    pct = count / len(test_predictions_original) * 100
    print(f"  ASA {asa}: {count} ({pct:.1f}%)")

# ============================================================
# 12. 方法比較
# ============================================================
print("\n\n【方法比較】")
print("=" * 70)

print(f"{'方法':<35} {'特徵數':>8} {'Kaggle Score':>15} {'本地 F1':>12}")
print("-" * 75)
print(f"{'A: Baseline':<35} {'13':>8} {'0.46513':>15} {'0.4517':>12}")
print(f"{'B: 完整特徵':<35} {'40':>8} {'0.52588':>15} {'0.5122':>12}")
print(f"{'C: 交互特徵 (degree=2)':<35} {'70':>8} {'0.53120':>15} {'0.4974':>12}")
print(f"{'E: 超參數調優':<35} {'70':>8} {'0.53809':>15} {'0.5002':>12}")
print(f"{'H: 縱向特徵 (本次)':<35} {X_train_final.shape[1]:>8} {'?':>15} {f1:>12.4f}")

print("\n" + "=" * 70)
print("方法 H: 縱向特徵工程完成!")
print("=" * 70)
print("\n請將 submission_longitudinal.csv 提交到 Kaggle 查看實際分數!")
