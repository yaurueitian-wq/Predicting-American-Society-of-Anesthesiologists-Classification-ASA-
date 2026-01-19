"""
方法 F: 方法C + SMOTE
使用 SMOTE 處理類別不平衡問題
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 載入並準備方法C的資料
# ============================================================
print("=" * 70)
print("方法 F: 方法C + SMOTE")
print("=" * 70)

# 載入方法B的特徵
train_b = pd.read_csv('../方法B_完整特徵/train_features.csv')
test_b = pd.read_csv('../方法B_完整特徵/test_features.csv')

target = 'ASA_Rating'
y_train = train_b[target].copy()

feature_cols_b = [col for col in train_b.columns if col != target]
X_train_b = train_b[feature_cols_b].copy()
X_test_b = test_b[feature_cols_b].copy()

categorical_features = ['Gender', 'ICU_Patient', 'Anesthesia_Method', 'Patient_Source',
                        'Surgery_Category', 'Drug_Category', 'Route_Standardized',
                        'Surgery_Procedure_Type']
numeric_features = [col for col in feature_cols_b if col not in categorical_features]

# 處理缺失值
for col in categorical_features:
    X_train_b[col] = X_train_b[col].fillna('Unknown')
    X_test_b[col] = X_test_b[col].fillna('Unknown')

for col in numeric_features:
    X_train_b[col] = X_train_b[col].fillna(0)
    X_test_b[col] = X_test_b[col].fillna(0)

# Label Encoding
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    all_values = pd.concat([X_train_b[col], X_test_b[col]]).unique()
    le.fit(all_values)
    X_train_b[col] = le.transform(X_train_b[col])
    X_test_b[col] = le.transform(X_test_b[col])
    label_encoders[col] = le

# 標準化
scaler = StandardScaler()
X_train_b[numeric_features] = scaler.fit_transform(X_train_b[numeric_features])
X_test_b[numeric_features] = scaler.transform(X_test_b[numeric_features])

# 0-based 標籤
y_train_encoded = y_train - 1

# ============================================================
# 2. 產生交互特徵 (方法C)
# ============================================================
print("\n【產生交互特徵】")

important_numeric = [
    'Age', 'BMI', 'Surgery_Count',
    'lab_abnormal_total', 'lab_count', 'lab_critical_HH',
    'med_count', 'catheter_count',
    'age_group', 'bmi_category',
    'is_elderly', 'is_obese'
]
important_numeric = [f for f in important_numeric if f in X_train_b.columns]

X_train_interact = X_train_b[important_numeric].copy()
X_test_interact = X_test_b[important_numeric].copy()

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_interact)
X_test_poly = poly.transform(X_test_interact)

poly_feature_names = poly.get_feature_names_out(important_numeric)
interaction_cols = [col for col in poly_feature_names if ' ' in col]

X_train_poly_df = pd.DataFrame(X_train_poly, columns=poly_feature_names, index=X_train_b.index)
X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly_feature_names, index=X_test_b.index)

X_train_interactions = X_train_poly_df[interaction_cols]
X_test_interactions = X_test_poly_df[interaction_cols]

# 特徵選擇
k_best = 30
selector = SelectKBest(score_func=f_classif, k=k_best)
selector.fit(X_train_interactions, y_train_encoded)
selected_mask = selector.get_support()
selected_interaction_cols = [col for col, selected in zip(interaction_cols, selected_mask) if selected]

# 合併成方法C資料
X_train_c = pd.concat([
    X_train_b.reset_index(drop=True),
    X_train_interactions[selected_interaction_cols].reset_index(drop=True)
], axis=1)

X_test_c = pd.concat([
    X_test_b.reset_index(drop=True),
    X_test_interactions[selected_interaction_cols].reset_index(drop=True)
], axis=1)

print(f"特徵數: {X_train_c.shape[1]}")

# ============================================================
# 3. 查看類別分布
# ============================================================
print("\n【原始類別分布】")
print("-" * 70)
class_dist = y_train_encoded.value_counts().sort_index()
for cls, count in class_dist.items():
    pct = count / len(y_train_encoded) * 100
    print(f"  ASA {cls+1}: {count} ({pct:.1f}%)")

# ============================================================
# 4. 切分驗證集
# ============================================================
print("\n【切分驗證集】")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_c, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)

print(f"訓練集: {X_tr.shape}")
print(f"驗證集: {X_val.shape}")

# ============================================================
# 5. 應用 SMOTE
# ============================================================
print("\n【應用 SMOTE】")
print("-" * 70)

smote = SMOTE(random_state=42)
X_tr_resampled, y_tr_resampled = smote.fit_resample(X_tr, y_tr)

print(f"SMOTE 前訓練集: {X_tr.shape[0]}")
print(f"SMOTE 後訓練集: {X_tr_resampled.shape[0]}")

print("\nSMOTE 後類別分布:")
resampled_dist = pd.Series(y_tr_resampled).value_counts().sort_index()
for cls, count in resampled_dist.items():
    pct = count / len(y_tr_resampled) * 100
    print(f"  ASA {cls+1}: {count} ({pct:.1f}%)")

# ============================================================
# 6. 訓練模型 (不使用 class_weight，因為已經用 SMOTE 平衡)
# ============================================================
print("\n【訓練模型】")
print("=" * 70)

model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)

model.fit(X_tr_resampled, y_tr_resampled)

# 驗證集評估
y_pred = model.predict(X_val)
f1 = f1_score(y_val, y_pred, average='macro')
acc = accuracy_score(y_val, y_pred)
kappa = cohen_kappa_score(y_val, y_pred)

print(f"\n驗證集 F1 Macro: {f1:.4f}")
print(f"驗證集 Accuracy: {acc:.4f}")
print(f"驗證集 Kappa: {kappa:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=['ASA 1', 'ASA 2', 'ASA 3', 'ASA 4']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_pred)
print(pd.DataFrame(cm,
                   index=['真實 ASA 1', '真實 ASA 2', '真實 ASA 3', '真實 ASA 4'],
                   columns=['預測 ASA 1', '預測 ASA 2', '預測 ASA 3', '預測 ASA 4']))

# ============================================================
# 7. 與不使用 SMOTE 比較
# ============================================================
print("\n\n【與不使用 SMOTE 比較】")
print("=" * 70)

# 不使用 SMOTE 的模型
model_no_smote = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    random_state=42,
    class_weight='balanced',
    verbose=-1,
    n_jobs=-1
)
model_no_smote.fit(X_tr, y_tr)
y_pred_no_smote = model_no_smote.predict(X_val)
f1_no_smote = f1_score(y_val, y_pred_no_smote, average='macro')

print(f"不使用 SMOTE (class_weight='balanced'): F1 = {f1_no_smote:.4f}")
print(f"使用 SMOTE:                            F1 = {f1:.4f}")
print(f"差異: {(f1 - f1_no_smote) / f1_no_smote * 100:+.2f}%")

# ============================================================
# 8. 使用完整訓練集 + SMOTE 重新訓練
# ============================================================
print("\n\n【測試集預測】")
print("=" * 70)

# 對完整訓練集應用 SMOTE
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_c, y_train_encoded)
print(f"完整訓練集 SMOTE 後: {X_train_resampled.shape[0]}")

# 訓練最終模型
final_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)
final_model.fit(X_train_resampled, y_train_resampled)

# 預測測試集
test_predictions = final_model.predict(X_test_c)
test_predictions_original = test_predictions + 1

# 輸出 Kaggle 格式
submission = pd.DataFrame({
    'Id': range(len(test_predictions)),
    'ASA_Rating': test_predictions_original
})

submission.to_csv('submission_smote.csv', index=False)
print(f"預測結果已儲存至 submission_smote.csv")

# 預測分布
print("\n測試集預測分布:")
pred_dist = pd.Series(test_predictions_original).value_counts().sort_index()
for asa, count in pred_dist.items():
    pct = count / len(test_predictions_original) * 100
    print(f"  ASA {asa}: {count} ({pct:.1f}%)")

# ============================================================
# 9. 方法比較
# ============================================================
print("\n\n【方法比較】")
print("=" * 70)

print(f"{'方法':<25} {'Kaggle Score':>15} {'本地 F1':>12}")
print("-" * 55)
print(f"{'A: Baseline':<25} {'0.46513':>15} {'0.4517':>12}")
print(f"{'B: 完整特徵':<25} {'0.52588':>15} {'0.5122':>12}")
print(f"{'C: 交互特徵':<25} {'0.53120':>15} {'0.4974':>12}")
print(f"{'D: 融合(B+C)':<25} {'0.52592':>15} {'0.5125':>12}")
print(f"{'E: 超參數調優':<25} {'0.53809':>15} {'0.5002':>12}")
print(f"{'F: SMOTE':<25} {'?':>15} {f1:>12.4f}")

print("\n" + "=" * 70)
print("方法 F: SMOTE 完成")
print("=" * 70)
