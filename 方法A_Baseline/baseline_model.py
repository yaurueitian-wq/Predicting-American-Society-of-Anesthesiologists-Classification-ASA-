"""
方法 A: Baseline 模型
使用現有 13 個特徵 (5 數值型 + 8 類別型) 建立 ASA 分級預測模型

特徵:
- 數值型: Age, HEIGHT, WEIGHT, BMI, Surgery_Count
- 類別型: Gender, ICU_Patient, Anesthesia_Method, Patient_Source,
         Surgery_Category, Drug_Category, Route_Standardized, Surgery_Procedure_Type
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import f1_score, cohen_kappa_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 載入資料
# ============================================================
print("=" * 70)
print("方法 A: Baseline 模型")
print("=" * 70)

train_df = pd.read_csv('../資料清洗/train_cleaned.csv')
test_df = pd.read_csv('../資料清洗/test_cleaned.csv')

print(f"\n訓練集: {train_df.shape}")
print(f"測試集: {test_df.shape}")

# ============================================================
# 2. 特徵選擇
# ============================================================
print("\n【特徵選擇】")

numeric_features = ['Age', 'HEIGHT', 'WEIGHT', 'BMI', 'Surgery_Count']
categorical_features = ['Gender', 'ICU_Patient', 'Anesthesia_Method', 'Patient_Source',
                        'Surgery_Category', 'Drug_Category', 'Route_Standardized',
                        'Surgery_Procedure_Type']

all_features = numeric_features + categorical_features
target = 'ASA_Rating'

print(f"數值型特徵: {numeric_features}")
print(f"類別型特徵: {categorical_features}")
print(f"總特徵數: {len(all_features)}")

# ============================================================
# 3. 資料前處理
# ============================================================
print("\n【資料前處理】")

# 複製資料
X_train = train_df[all_features].copy()
y_train = train_df[target].copy()
X_test = test_df[all_features].copy()

# XGBoost 需要 0-based 標籤，將 ASA 1-4 轉為 0-3
y_train = y_train - 1

# 處理缺失值 (Anesthesia_Method 有 11 筆缺失)
for col in categorical_features:
    X_train[col] = X_train[col].fillna('Unknown')
    X_test[col] = X_test[col].fillna('Unknown')

# Label Encoding for categorical features
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    # 合併訓練集和測試集的類別進行 fit
    all_values = pd.concat([X_train[col], X_test[col]]).unique()
    le.fit(all_values)
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"缺失值: {X_train.isna().sum().sum()}")

# 標準化數值特徵
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# ============================================================
# 4. 切分驗證集
# ============================================================
print("\n【切分驗證集】")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"訓練集: {X_tr.shape}")
print(f"驗證集: {X_val.shape}")

# ============================================================
# 5. 建立多個模型
# ============================================================
print("\n【模型訓練與評估】")
print("=" * 70)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False,
                                  eval_metric='mlogloss', n_jobs=-1),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced',
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
# 6. 結果比較
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
# 7. 最佳模型詳細報告
# ============================================================
print("\n\n【最佳模型詳細報告】")
print("=" * 70)

y_pred_best = best_model.predict(X_val)

print("\nClassification Report:")
print(classification_report(y_val, y_pred_best, target_names=['ASA 1', 'ASA 2', 'ASA 3', 'ASA 4']))

# 將預測結果轉回 1-4
y_pred_best_display = y_pred_best + 1
y_val_display = y_val + 1

print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_pred_best)
print(pd.DataFrame(cm,
                   index=['真實 ASA 1', '真實 ASA 2', '真實 ASA 3', '真實 ASA 4'],
                   columns=['預測 ASA 1', '預測 ASA 2', '預測 ASA 3', '預測 ASA 4']))

# ============================================================
# 8. 特徵重要性 (如果模型支援)
# ============================================================
if hasattr(best_model, 'feature_importances_'):
    print("\n\n【特徵重要性】")
    print("=" * 70)

    importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    for _, row in importance_df.iterrows():
        bar = '█' * int(row['Importance'] * 50)
        print(f"  {row['Feature']:<25} {row['Importance']:.4f} {bar}")

# ============================================================
# 9. 使用完整訓練集重新訓練並預測測試集
# ============================================================
print("\n\n【測試集預測】")
print("=" * 70)

# 使用完整訓練集重新訓練最佳模型
best_model.fit(X_train, y_train)

# 預測測試集
test_predictions = best_model.predict(X_test)

# 將預測結果轉回 1-4 (原始 ASA 分級)
test_predictions_original = test_predictions + 1

# 輸出預測結果 (符合 Kaggle 格式: Id, ASA_Rating)
submission = pd.DataFrame({
    'Id': range(len(test_predictions)),
    'ASA_Rating': test_predictions_original
})

submission.to_csv('submission_baseline.csv', index=False)
print(f"預測結果已儲存至 submission_baseline.csv")

# 預測分布
print("\n測試集預測分布:")
pred_dist = pd.Series(test_predictions_original).value_counts().sort_index()
for asa, count in pred_dist.items():
    pct = count / len(test_predictions_original) * 100
    print(f"  ASA {asa}: {count} ({pct:.1f}%)")

print("\n" + "=" * 70)
print("方法 A: Baseline 模型完成")
print("=" * 70)
