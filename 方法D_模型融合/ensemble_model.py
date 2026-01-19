"""
方法 D: 模型融合
結合方法B (完整特徵) 和方法C (交互特徵) 的預測結果
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 載入資料
# ============================================================
print("=" * 70)
print("方法 D: 模型融合 (B + C)")
print("=" * 70)

# 載入方法B的特徵
train_b = pd.read_csv('../方法B_完整特徵/train_features.csv')
test_b = pd.read_csv('../方法B_完整特徵/test_features.csv')

print(f"\n方法B特徵: {train_b.shape[1] - 1} 個")

# ============================================================
# 2. 準備方法B的資料
# ============================================================
print("\n【準備方法B資料】")

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
scaler_b = StandardScaler()
X_train_b[numeric_features] = scaler_b.fit_transform(X_train_b[numeric_features])
X_test_b[numeric_features] = scaler_b.transform(X_test_b[numeric_features])

# 0-based 標籤
y_train_encoded = y_train - 1

print(f"X_train_b shape: {X_train_b.shape}")

# ============================================================
# 3. 準備方法C的交互特徵
# ============================================================
print("\n【準備方法C交互特徵】")

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

print(f"X_train_c shape: {X_train_c.shape}")

# ============================================================
# 4. 切分驗證集
# ============================================================
print("\n【切分驗證集】")

X_tr_b, X_val_b, y_tr, y_val = train_test_split(
    X_train_b, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)

X_tr_c = X_train_c.iloc[X_tr_b.index].reset_index(drop=True)
X_val_c = X_train_c.iloc[X_val_b.index].reset_index(drop=True)
X_tr_b = X_tr_b.reset_index(drop=True)
X_val_b = X_val_b.reset_index(drop=True)

print(f"訓練集: {X_tr_b.shape[0]}")
print(f"驗證集: {X_val_b.shape[0]}")

# ============================================================
# 5. 訓練模型B和C
# ============================================================
print("\n【訓練基礎模型】")
print("-" * 70)

# 模型B: LightGBM on 方法B特徵
model_b = lgb.LGBMClassifier(
    n_estimators=200, max_depth=10, learning_rate=0.1,
    random_state=42, class_weight='balanced', verbose=-1, n_jobs=-1
)
model_b.fit(X_tr_b, y_tr)
y_pred_b = model_b.predict(X_val_b)
y_proba_b = model_b.predict_proba(X_val_b)

f1_b = f1_score(y_val, y_pred_b, average='macro')
print(f"模型B (40特徵) - F1 Macro: {f1_b:.4f}")

# 模型C: LightGBM on 方法C特徵
model_c = lgb.LGBMClassifier(
    n_estimators=200, max_depth=10, learning_rate=0.1,
    random_state=42, class_weight='balanced', verbose=-1, n_jobs=-1
)
model_c.fit(X_tr_c, y_tr)
y_pred_c = model_c.predict(X_val_c)
y_proba_c = model_c.predict_proba(X_val_c)

f1_c = f1_score(y_val, y_pred_c, average='macro')
print(f"模型C (70特徵) - F1 Macro: {f1_c:.4f}")

# ============================================================
# 6. 嘗試不同的融合策略
# ============================================================
print("\n【融合策略比較】")
print("=" * 70)

results = []

# 策略1: 硬投票 (Hard Voting)
y_pred_vote = np.where(y_pred_b == y_pred_c, y_pred_b,
                       np.where(y_proba_b.max(axis=1) > y_proba_c.max(axis=1), y_pred_b, y_pred_c))
f1_vote = f1_score(y_val, y_pred_vote, average='macro')
acc_vote = accuracy_score(y_val, y_pred_vote)
print(f"1. 硬投票 (信心度決定) - F1: {f1_vote:.4f}, Acc: {acc_vote:.4f}")
results.append(('硬投票', f1_vote, acc_vote))

# 策略2: 軟投票 - 平均機率
y_proba_avg = (y_proba_b + y_proba_c) / 2
y_pred_avg = y_proba_avg.argmax(axis=1)
f1_avg = f1_score(y_val, y_pred_avg, average='macro')
acc_avg = accuracy_score(y_val, y_pred_avg)
print(f"2. 軟投票 (平均機率) - F1: {f1_avg:.4f}, Acc: {acc_avg:.4f}")
results.append(('軟投票-平均', f1_avg, acc_avg))

# 策略3-7: 不同權重的加權平均
best_weight = 0.5
best_f1 = 0
for w_b in [0.3, 0.4, 0.5, 0.6, 0.7]:
    w_c = 1 - w_b
    y_proba_weighted = w_b * y_proba_b + w_c * y_proba_c
    y_pred_weighted = y_proba_weighted.argmax(axis=1)
    f1_weighted = f1_score(y_val, y_pred_weighted, average='macro')
    acc_weighted = accuracy_score(y_val, y_pred_weighted)
    print(f"3. 加權 (B:{w_b:.1f}, C:{w_c:.1f}) - F1: {f1_weighted:.4f}, Acc: {acc_weighted:.4f}")
    results.append((f'加權-B{w_b}', f1_weighted, acc_weighted))
    if f1_weighted > best_f1:
        best_f1 = f1_weighted
        best_weight = w_b

print(f"\n最佳權重: B={best_weight:.1f}, C={1-best_weight:.1f}")

# ============================================================
# 7. 使用最佳融合策略
# ============================================================
print("\n\n【最佳融合模型詳細報告】")
print("=" * 70)

# 使用最佳權重
w_b = best_weight
w_c = 1 - w_b
y_proba_best = w_b * y_proba_b + w_c * y_proba_c
y_pred_best = y_proba_best.argmax(axis=1)

print(f"\n使用權重: B={w_b:.1f}, C={w_c:.1f}")
print(f"F1 Macro: {f1_score(y_val, y_pred_best, average='macro'):.4f}")
print(f"Accuracy: {accuracy_score(y_val, y_pred_best):.4f}")
print(f"Cohen's Kappa: {cohen_kappa_score(y_val, y_pred_best):.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_pred_best, target_names=['ASA 1', 'ASA 2', 'ASA 3', 'ASA 4']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_pred_best)
print(pd.DataFrame(cm,
                   index=['真實 ASA 1', '真實 ASA 2', '真實 ASA 3', '真實 ASA 4'],
                   columns=['預測 ASA 1', '預測 ASA 2', '預測 ASA 3', '預測 ASA 4']))

# ============================================================
# 8. 使用完整訓練集重新訓練並預測測試集
# ============================================================
print("\n\n【測試集預測】")
print("=" * 70)

# 重新訓練模型B
model_b_full = lgb.LGBMClassifier(
    n_estimators=200, max_depth=10, learning_rate=0.1,
    random_state=42, class_weight='balanced', verbose=-1, n_jobs=-1
)
model_b_full.fit(X_train_b, y_train_encoded)
test_proba_b = model_b_full.predict_proba(X_test_b)

# 重新訓練模型C
model_c_full = lgb.LGBMClassifier(
    n_estimators=200, max_depth=10, learning_rate=0.1,
    random_state=42, class_weight='balanced', verbose=-1, n_jobs=-1
)
model_c_full.fit(X_train_c, y_train_encoded)
test_proba_c = model_c_full.predict_proba(X_test_c)

# 融合預測
test_proba_ensemble = w_b * test_proba_b + w_c * test_proba_c
test_predictions = test_proba_ensemble.argmax(axis=1)

# 轉回 1-4
test_predictions_original = test_predictions + 1

# 輸出 Kaggle 格式
submission = pd.DataFrame({
    'Id': range(len(test_predictions)),
    'ASA_Rating': test_predictions_original
})

submission.to_csv('submission_ensemble.csv', index=False)
print(f"預測結果已儲存至 submission_ensemble.csv")

# 預測分布
print("\n測試集預測分布:")
pred_dist = pd.Series(test_predictions_original).value_counts().sort_index()
for asa, count in pred_dist.items():
    pct = count / len(test_predictions_original) * 100
    print(f"  ASA {asa}: {count} ({pct:.1f}%)")

# ============================================================
# 9. 四種方法比較
# ============================================================
print("\n\n【四種方法比較】")
print("=" * 70)

comparison = {
    '方法': ['A: Baseline', 'B: 完整特徵', 'C: 交互特徵', 'D: 融合(B+C)'],
    '特徵數': [13, 40, 70, '40+70'],
    'Kaggle Score': [0.46513, 0.52588, 0.53120, '?'],
    '本地 F1': [0.4517, 0.5122, 0.4974, f1_score(y_val, y_pred_best, average='macro')]
}

print(f"{'方法':<20} {'特徵數':>10} {'Kaggle Score':>15} {'本地 F1':>12}")
print("-" * 60)
for i in range(4):
    kaggle = comparison['Kaggle Score'][i]
    kaggle_str = f"{kaggle:.5f}" if isinstance(kaggle, float) else kaggle
    local_f1 = comparison['本地 F1'][i]
    local_str = f"{local_f1:.4f}" if isinstance(local_f1, float) else local_f1
    print(f"{comparison['方法'][i]:<20} {str(comparison['特徵數'][i]):>10} {kaggle_str:>15} {local_str:>12}")

print("\n" + "=" * 70)
print("方法 D: 模型融合完成")
print("=" * 70)
