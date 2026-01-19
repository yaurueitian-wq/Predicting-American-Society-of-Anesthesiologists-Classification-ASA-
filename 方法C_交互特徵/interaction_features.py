"""
方法 C: 交互特徵模型
使用 PolynomialFeatures 自動產生二階交互特徵
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 載入方法B的特徵工程資料
# ============================================================
print("=" * 70)
print("方法 C: 交互特徵模型")
print("=" * 70)

train_df = pd.read_csv('../方法B_完整特徵/train_features.csv')
test_df = pd.read_csv('../方法B_完整特徵/test_features.csv')

print(f"\n原始訓練集: {train_df.shape}")
print(f"原始測試集: {test_df.shape}")

# ============================================================
# 2. 準備特徵與目標
# ============================================================
print("\n【準備特徵與目標】")

target = 'ASA_Rating'
y_train = train_df[target].copy()

# 特徵 (排除目標變數)
feature_cols = [col for col in train_df.columns if col != target]
X_train = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()

# 識別數值型和類別型特徵
categorical_features = ['Gender', 'ICU_Patient', 'Anesthesia_Method', 'Patient_Source',
                        'Surgery_Category', 'Drug_Category', 'Route_Standardized',
                        'Surgery_Procedure_Type']
numeric_features = [col for col in feature_cols if col not in categorical_features]

print(f"數值型特徵: {len(numeric_features)}")
print(f"類別型特徵: {len(categorical_features)}")

# ============================================================
# 3. 資料前處理
# ============================================================
print("\n【資料前處理】")

# 處理缺失值
for col in categorical_features:
    X_train[col] = X_train[col].fillna('Unknown')
    X_test[col] = X_test[col].fillna('Unknown')

for col in numeric_features:
    X_train[col] = X_train[col].fillna(0)
    X_test[col] = X_test[col].fillna(0)

# Label Encoding for categorical features
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    all_values = pd.concat([X_train[col], X_test[col]]).unique()
    le.fit(all_values)
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

# XGBoost 需要 0-based 標籤
y_train = y_train - 1

# 標準化數值特徵
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

print(f"前處理後 X_train shape: {X_train.shape}")
print(f"前處理後 X_test shape: {X_test.shape}")

# ============================================================
# 4. 產生交互特徵 (PolynomialFeatures)
# ============================================================
print("\n【產生交互特徵】")
print("-" * 70)

# 選擇用於交互的重要數值特徵 (避免維度爆炸)
# 根據方法B的特徵重要性，選擇最重要的數值特徵
important_numeric = [
    'Age', 'BMI', 'Surgery_Count',
    'lab_abnormal_total', 'lab_count', 'lab_critical_HH',
    'med_count', 'catheter_count',
    'age_group', 'bmi_category',
    'is_elderly', 'is_obese'
]

# 確保所有特徵都存在
important_numeric = [f for f in important_numeric if f in X_train.columns]
print(f"用於交互的特徵: {important_numeric}")

# 提取用於交互的特徵
X_train_interact = X_train[important_numeric].copy()
X_test_interact = X_test[important_numeric].copy()

# 使用 PolynomialFeatures 產生二階交互
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_interact)
X_test_poly = poly.transform(X_test_interact)

# 獲取交互特徵名稱
poly_feature_names = poly.get_feature_names_out(important_numeric)
print(f"交互特徵數量: {len(poly_feature_names)}")
print(f"原始特徵: {len(important_numeric)}")
print(f"新增交互特徵: {len(poly_feature_names) - len(important_numeric)}")

# 轉換為 DataFrame
X_train_poly_df = pd.DataFrame(X_train_poly, columns=poly_feature_names, index=X_train.index)
X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly_feature_names, index=X_test.index)

# 只保留交互項 (排除原始特徵，因為已經在 X_train 中)
interaction_cols = [col for col in poly_feature_names if ' ' in col]
X_train_interactions = X_train_poly_df[interaction_cols]
X_test_interactions = X_test_poly_df[interaction_cols]

print(f"純交互特徵數量: {len(interaction_cols)}")

# ============================================================
# 5. 特徵選擇 - 篩選最重要的交互特徵
# ============================================================
print("\n【特徵選擇 - 篩選重要交互特徵】")
print("-" * 70)

# 使用 SelectKBest 選擇最重要的交互特徵
k_best = min(30, len(interaction_cols))  # 最多選擇 30 個交互特徵
selector = SelectKBest(score_func=f_classif, k=k_best)
selector.fit(X_train_interactions, y_train)

# 獲取選中的特徵
selected_mask = selector.get_support()
selected_interaction_cols = [col for col, selected in zip(interaction_cols, selected_mask) if selected]

print(f"選擇的交互特徵數量: {len(selected_interaction_cols)}")

# 顯示 Top 10 交互特徵及其分數
scores = pd.DataFrame({
    'Feature': interaction_cols,
    'Score': selector.scores_
}).sort_values('Score', ascending=False)

print("\nTop 10 交互特徵:")
for i, (_, row) in enumerate(scores.head(10).iterrows()):
    print(f"  {i+1:2}. {row['Feature']:<35} Score: {row['Score']:.2f}")

# ============================================================
# 6. 合併原始特徵與選擇的交互特徵
# ============================================================
print("\n【合併特徵】")
print("-" * 70)

# 合併原始特徵與選擇的交互特徵
X_train_final = pd.concat([
    X_train.reset_index(drop=True),
    X_train_interactions[selected_interaction_cols].reset_index(drop=True)
], axis=1)

X_test_final = pd.concat([
    X_test.reset_index(drop=True),
    X_test_interactions[selected_interaction_cols].reset_index(drop=True)
], axis=1)

print(f"最終訓練集 shape: {X_train_final.shape}")
print(f"最終測試集 shape: {X_test_final.shape}")
print(f"原始特徵: {len(feature_cols)}")
print(f"新增交互特徵: {len(selected_interaction_cols)}")
print(f"總特徵數: {X_train_final.shape[1]}")

# ============================================================
# 7. 儲存交互特徵資料
# ============================================================
print("\n【儲存交互特徵資料】")
print("-" * 70)

# 加入目標變數並儲存
train_final = X_train_final.copy()
train_final['ASA_Rating'] = y_train.values + 1  # 轉回 1-4

train_final.to_csv('train_interaction_features.csv', index=False)
X_test_final.to_csv('test_interaction_features.csv', index=False)

print(f"已儲存 train_interaction_features.csv ({train_final.shape})")
print(f"已儲存 test_interaction_features.csv ({X_test_final.shape})")

# ============================================================
# 8. 切分驗證集
# ============================================================
print("\n【切分驗證集】")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"訓練集: {X_tr.shape}")
print(f"驗證集: {X_val.shape}")

# ============================================================
# 9. 建立多個模型
# ============================================================
print("\n【模型訓練與評估】")
print("=" * 70)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                            max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  random_state=42, use_label_encoder=False,
                                  eval_metric='mlogloss', n_jobs=-1),
    'LightGBM': lgb.LGBMClassifier(n_estimators=200, max_depth=10, learning_rate=0.1,
                                    random_state=42, class_weight='balanced',
                                    verbose=-1, n_jobs=-1)
}

results = []

for name, model in models.items():
    print(f"\n--- {name} ---")

    # 訓練
    model.fit(X_tr, y_tr)

    # 預測
    y_pred = model.predict(X_val)

    # 評估指標
    accuracy = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    f1_weighted = f1_score(y_val, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_val, y_pred)

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 (Macro): {f1_macro:.4f}")
    print(f"  F1 (Weighted): {f1_weighted:.4f}")
    print(f"  Cohen's Kappa: {kappa:.4f}")

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'F1_Macro': f1_macro,
        'F1_Weighted': f1_weighted,
        'Kappa': kappa
    })

# ============================================================
# 10. 結果比較
# ============================================================
print("\n\n【模型比較結果】")
print("=" * 70)

results_df = pd.DataFrame(results).sort_values('F1_Macro', ascending=False)
print(results_df.to_string(index=False))

# 最佳模型
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
print(f"\n最佳模型: {best_model_name}")

# ============================================================
# 11. 最佳模型詳細報告
# ============================================================
print("\n\n【最佳模型詳細報告】")
print("=" * 70)

y_pred_best = best_model.predict(X_val)

print("\nClassification Report:")
print(classification_report(y_val, y_pred_best, target_names=['ASA 1', 'ASA 2', 'ASA 3', 'ASA 4']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_pred_best)
print(pd.DataFrame(cm,
                   index=['真實 ASA 1', '真實 ASA 2', '真實 ASA 3', '真實 ASA 4'],
                   columns=['預測 ASA 1', '預測 ASA 2', '預測 ASA 3', '預測 ASA 4']))

# ============================================================
# 12. 特徵重要性
# ============================================================
if hasattr(best_model, 'feature_importances_'):
    print("\n\n【特徵重要性 Top 25】")
    print("=" * 70)

    all_feature_names = list(X_train_final.columns)
    importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # 區分原始特徵和交互特徵
    print("\n原始特徵:")
    orig_importance = importance_df[~importance_df['Feature'].str.contains(' ')]
    for i, (_, row) in enumerate(orig_importance.head(15).iterrows()):
        bar = '█' * int(row['Importance'] * 100)
        print(f"  {i+1:2}. {row['Feature']:<25} {row['Importance']:.4f} {bar}")

    print("\n交互特徵:")
    inter_importance = importance_df[importance_df['Feature'].str.contains(' ')]
    for i, (_, row) in enumerate(inter_importance.head(10).iterrows()):
        bar = '█' * int(row['Importance'] * 100)
        print(f"  {i+1:2}. {row['Feature']:<35} {row['Importance']:.4f} {bar}")

# ============================================================
# 13. 交叉驗證
# ============================================================
print("\n\n【5-Fold 交叉驗證】")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_train_final, y_train, cv=cv, scoring='f1_macro')

print(f"  F1 Macro scores: {cv_scores}")
print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================
# 14. 使用完整訓練集重新訓練並預測測試集
# ============================================================
print("\n\n【測試集預測】")
print("=" * 70)

# 使用完整訓練集重新訓練最佳模型
best_model.fit(X_train_final, y_train)

# 預測測試集
test_predictions = best_model.predict(X_test_final)

# 將預測結果轉回 1-4 (原始 ASA 分級)
test_predictions_original = test_predictions + 1

# 輸出預測結果 (符合 Kaggle 格式)
submission = pd.DataFrame({
    'Id': range(len(test_predictions)),
    'ASA_Rating': test_predictions_original
})

submission.to_csv('submission_interaction.csv', index=False)
print(f"預測結果已儲存至 submission_interaction.csv")

# 預測分布
print("\n測試集預測分布:")
pred_dist = pd.Series(test_predictions_original).value_counts().sort_index()
for asa, count in pred_dist.items():
    pct = count / len(test_predictions_original) * 100
    print(f"  ASA {asa}: {count} ({pct:.1f}%)")

# ============================================================
# 15. 三種方法比較
# ============================================================
print("\n\n【三種方法比較】")
print("=" * 70)

comparison = {
    '方法': ['A: Baseline (13特徵)', 'B: 完整特徵 (40特徵)', 'C: 交互特徵'],
    '特徵數': [13, 40, X_train_final.shape[1]],
    'F1_Macro': [0.4517, 0.5122, results_df.iloc[0]['F1_Macro']],
    'Accuracy': [0.4927, 0.5748, results_df.iloc[0]['Accuracy']]
}

comparison_df = pd.DataFrame(comparison)
print(comparison_df.to_string(index=False))

# 改進幅度
print("\n改進幅度 (相對於 Baseline):")
baseline_f1 = 0.4517
advanced_f1 = 0.5122
interaction_f1 = results_df.iloc[0]['F1_Macro']

print(f"  方法 B vs A: {(advanced_f1 - baseline_f1) / baseline_f1 * 100:+.1f}%")
print(f"  方法 C vs A: {(interaction_f1 - baseline_f1) / baseline_f1 * 100:+.1f}%")
print(f"  方法 C vs B: {(interaction_f1 - advanced_f1) / advanced_f1 * 100:+.1f}%")

print("\n" + "=" * 70)
print("方法 C: 交互特徵模型完成")
print("=" * 70)
