"""
方法 G: 三階交互特徵模型
使用 PolynomialFeatures (degree=3) 產生三階交互特徵
基於方法E的最佳超參數
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 載入方法B的特徵工程資料
# ============================================================
print("=" * 70)
print("方法 G: 三階交互特徵模型")
print("=" * 70)

train_b = pd.read_csv('../方法B_完整特徵/train_features.csv')
test_b = pd.read_csv('../方法B_完整特徵/test_features.csv')

print(f"\n原始訓練集: {train_b.shape}")
print(f"原始測試集: {test_b.shape}")

target = 'ASA_Rating'
y_train = train_b[target].copy()

feature_cols_b = [col for col in train_b.columns if col != target]
X_train_b = train_b[feature_cols_b].copy()
X_test_b = test_b[feature_cols_b].copy()

# ============================================================
# 2. 資料前處理
# ============================================================
print("\n【資料前處理】")

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

print(f"前處理後 X_train shape: {X_train_b.shape}")

# ============================================================
# 3. 產生三階交互特徵 (degree=3)
# ============================================================
print("\n【產生三階交互特徵】")
print("-" * 70)

# 選擇用於交互的重要數值特徵 (減少特徵數以避免維度爆炸)
# 三階交互會產生 C(n,3) + C(n,2) 個交互項，需謹慎選擇
important_numeric = [
    'Age', 'BMI', 'Surgery_Count',
    'lab_abnormal_total', 'lab_count',
    'med_count', 'catheter_count',
    'is_elderly', 'is_obese'
]
important_numeric = [f for f in important_numeric if f in X_train_b.columns]

print(f"用於三階交互的特徵 ({len(important_numeric)} 個):")
for f in important_numeric:
    print(f"  - {f}")

# 提取用於交互的特徵
X_train_interact = X_train_b[important_numeric].copy()
X_test_interact = X_test_b[important_numeric].copy()

# 使用 PolynomialFeatures 產生三階交互 (degree=3)
print("\n產生 degree=3 交互特徵...")
poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_interact)
X_test_poly = poly.transform(X_test_interact)

# 獲取交互特徵名稱
poly_feature_names = poly.get_feature_names_out(important_numeric)
print(f"總交互特徵數量: {len(poly_feature_names)}")

# 分類：原始特徵、二階交互、三階交互
original_cols = [col for col in poly_feature_names if ' ' not in col]
degree2_cols = [col for col in poly_feature_names if col.count(' ') == 1]
degree3_cols = [col for col in poly_feature_names if col.count(' ') == 2]

print(f"  - 原始特徵: {len(original_cols)}")
print(f"  - 二階交互: {len(degree2_cols)}")
print(f"  - 三階交互: {len(degree3_cols)}")

# 轉換為 DataFrame
X_train_poly_df = pd.DataFrame(X_train_poly, columns=poly_feature_names, index=X_train_b.index)
X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly_feature_names, index=X_test_b.index)

# 只保留交互項 (二階+三階)
interaction_cols = degree2_cols + degree3_cols
X_train_interactions = X_train_poly_df[interaction_cols]
X_test_interactions = X_test_poly_df[interaction_cols]

print(f"\n純交互特徵數量 (二階+三階): {len(interaction_cols)}")

# ============================================================
# 4. 特徵選擇 - 篩選最重要的交互特徵
# ============================================================
print("\n【特徵選擇 - 篩選重要交互特徵】")
print("-" * 70)

# 使用 SelectKBest 選擇最重要的交互特徵
k_best = min(50, len(interaction_cols))  # 最多選擇 50 個交互特徵
selector = SelectKBest(score_func=f_classif, k=k_best)
selector.fit(X_train_interactions, y_train_encoded)

# 獲取選中的特徵
selected_mask = selector.get_support()
selected_interaction_cols = [col for col, selected in zip(interaction_cols, selected_mask) if selected]

print(f"選擇的交互特徵數量: {len(selected_interaction_cols)}")

# 統計選中的二階和三階特徵
selected_degree2 = [col for col in selected_interaction_cols if col.count(' ') == 1]
selected_degree3 = [col for col in selected_interaction_cols if col.count(' ') == 2]
print(f"  - 選中的二階特徵: {len(selected_degree2)}")
print(f"  - 選中的三階特徵: {len(selected_degree3)}")

# 顯示 Top 15 交互特徵及其分數
scores = pd.DataFrame({
    'Feature': interaction_cols,
    'Score': selector.scores_
}).sort_values('Score', ascending=False)

print("\nTop 15 交互特徵:")
for i, (_, row) in enumerate(scores.head(15).iterrows()):
    degree = "3階" if row['Feature'].count(' ') == 2 else "2階"
    print(f"  {i+1:2}. [{degree}] {row['Feature']:<40} Score: {row['Score']:.2f}")

# ============================================================
# 5. 合併原始特徵與選擇的交互特徵
# ============================================================
print("\n【合併特徵】")
print("-" * 70)

# 合併原始特徵與選擇的交互特徵
X_train_g = pd.concat([
    X_train_b.reset_index(drop=True),
    X_train_interactions[selected_interaction_cols].reset_index(drop=True)
], axis=1)

X_test_g = pd.concat([
    X_test_b.reset_index(drop=True),
    X_test_interactions[selected_interaction_cols].reset_index(drop=True)
], axis=1)

print(f"最終訓練集 shape: {X_train_g.shape}")
print(f"最終測試集 shape: {X_test_g.shape}")
print(f"原始特徵: {len(feature_cols_b)}")
print(f"新增交互特徵: {len(selected_interaction_cols)}")
print(f"總特徵數: {X_train_g.shape[1]}")

# ============================================================
# 6. 儲存三階交互特徵資料集
# ============================================================
print("\n【儲存三階交互特徵資料集】")
print("-" * 70)

# 加入目標變數並儲存
train_g = X_train_g.copy()
train_g['ASA_Rating'] = y_train.values

train_g.to_csv('train_degree3_features.csv', index=False)
X_test_g.to_csv('test_degree3_features.csv', index=False)

print(f"已儲存 train_degree3_features.csv ({train_g.shape})")
print(f"已儲存 test_degree3_features.csv ({X_test_g.shape})")

# 儲存選中的交互特徵列表
with open('selected_interaction_features.txt', 'w') as f:
    f.write("# 方法G: 選中的交互特徵\n")
    f.write(f"# 總共 {len(selected_interaction_cols)} 個\n\n")
    f.write("## 二階交互特徵:\n")
    for col in selected_degree2:
        f.write(f"  - {col}\n")
    f.write("\n## 三階交互特徵:\n")
    for col in selected_degree3:
        f.write(f"  - {col}\n")

print(f"已儲存 selected_interaction_features.txt")

# ============================================================
# 7. 切分驗證集
# ============================================================
print("\n【切分驗證集】")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_g, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)

print(f"訓練集: {X_tr.shape}")
print(f"驗證集: {X_val.shape}")

# ============================================================
# 8. 使用方法E的最佳超參數訓練模型
# ============================================================
print("\n【使用方法E最佳超參數訓練模型】")
print("=" * 70)

# 方法E找到的最佳超參數
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

print("使用超參數:")
for key, value in best_params.items():
    if key not in ['random_state', 'verbose', 'n_jobs']:
        print(f"  {key}: {value}")

# 訓練模型
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

print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_pred)
print(pd.DataFrame(cm,
                   index=['真實 ASA 1', '真實 ASA 2', '真實 ASA 3', '真實 ASA 4'],
                   columns=['預測 ASA 1', '預測 ASA 2', '預測 ASA 3', '預測 ASA 4']))

# ============================================================
# 9. 特徵重要性分析
# ============================================================
print("\n【特徵重要性 Top 25】")
print("=" * 70)

all_feature_names = list(X_train_g.columns)
importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n原始特徵 Top 10:")
orig_importance = importance_df[~importance_df['Feature'].str.contains(' ')]
for i, (_, row) in enumerate(orig_importance.head(10).iterrows()):
    bar = '█' * int(row['Importance'] * 100)
    print(f"  {i+1:2}. {row['Feature']:<25} {row['Importance']:.4f} {bar}")

print("\n交互特徵 Top 15:")
inter_importance = importance_df[importance_df['Feature'].str.contains(' ')]
for i, (_, row) in enumerate(inter_importance.head(15).iterrows()):
    degree = "3階" if row['Feature'].count(' ') == 2 else "2階"
    bar = '█' * int(row['Importance'] * 100)
    print(f"  {i+1:2}. [{degree}] {row['Feature']:<40} {row['Importance']:.4f} {bar}")

# ============================================================
# 10. 5-Fold 交叉驗證
# ============================================================
print("\n\n【5-Fold 交叉驗證】")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    lgb.LGBMClassifier(**best_params),
    X_train_g, y_train_encoded, cv=cv, scoring='f1_macro'
)

print(f"F1 Macro scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================
# 11. 使用完整訓練集重新訓練並預測測試集
# ============================================================
print("\n\n【測試集預測】")
print("=" * 70)

# 使用完整訓練集訓練
final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X_train_g, y_train_encoded)

# 預測測試集
test_predictions = final_model.predict(X_test_g)
test_predictions_original = test_predictions + 1

# 輸出 Kaggle 格式
submission = pd.DataFrame({
    'Id': range(len(test_predictions)),
    'ASA_Rating': test_predictions_original
})

submission.to_csv('submission_degree3.csv', index=False)
print(f"預測結果已儲存至 submission_degree3.csv")

# 預測分布
print("\n測試集預測分布:")
pred_dist = pd.Series(test_predictions_original).value_counts().sort_index()
for asa, count in pred_dist.items():
    pct = count / len(test_predictions_original) * 100
    print(f"  ASA {asa}: {count} ({pct:.1f}%)")

# ============================================================
# 12. 與其他方法比較
# ============================================================
print("\n\n【方法比較】")
print("=" * 70)

print(f"{'方法':<30} {'特徵數':>8} {'Kaggle Score':>15} {'本地 F1':>12}")
print("-" * 70)
print(f"{'A: Baseline':<30} {'13':>8} {'0.46513':>15} {'0.4517':>12}")
print(f"{'B: 完整特徵':<30} {'40':>8} {'0.52588':>15} {'0.5122':>12}")
print(f"{'C: 交互特徵 (degree=2)':<30} {'70':>8} {'0.53120':>15} {'0.4974':>12}")
print(f"{'E: 超參數調優':<30} {'70':>8} {'0.53809':>15} {'0.5002':>12}")
print(f"{'G: 三階交互 (degree=3)':<30} {X_train_g.shape[1]:>8} {'?':>15} {f1:>12.4f}")

print("\n" + "=" * 70)
print("方法 G: 三階交互特徵模型完成")
print("=" * 70)
print("\n請將 submission_degree3.csv 提交到 Kaggle 查看實際分數！")
