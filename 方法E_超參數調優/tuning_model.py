"""
方法 E: 基於方法C的超參數調優
使用 Optuna 對 LightGBM 進行超參數搜索
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# 1. 載入並準備方法C的資料
# ============================================================
print("=" * 70)
print("方法 E: 基於方法C的超參數調優")
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
# 3. 切分驗證集
# ============================================================
print("\n【切分驗證集】")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_c, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)

print(f"訓練集: {X_tr.shape}")
print(f"驗證集: {X_val.shape}")

# ============================================================
# 4. Optuna 超參數調優
# ============================================================
print("\n【Optuna 超參數調優】")
print("=" * 70)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'class_weight': 'balanced',
        'verbose': -1,
        'n_jobs': -1
    }

    model = lgb.LGBMClassifier(**params)

    # 使用 3-fold CV
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring='f1_macro', n_jobs=-1)

    return scores.mean()

# 執行調優
print("開始搜索最佳超參數 (100 trials)...")
sampler = TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"\n最佳 F1 Macro (CV): {study.best_value:.4f}")
print("\n最佳超參數:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# ============================================================
# 5. 使用最佳超參數訓練模型
# ============================================================
print("\n\n【使用最佳超參數訓練模型】")
print("=" * 70)

best_params = study.best_params
best_params['random_state'] = 42
best_params['class_weight'] = 'balanced'
best_params['verbose'] = -1
best_params['n_jobs'] = -1

# 訓練最佳模型
best_model = lgb.LGBMClassifier(**best_params)
best_model.fit(X_tr, y_tr)

# 驗證集評估
y_pred = best_model.predict(X_val)
f1 = f1_score(y_val, y_pred, average='macro')
acc = accuracy_score(y_val, y_pred)
kappa = cohen_kappa_score(y_val, y_pred)

print(f"驗證集 F1 Macro: {f1:.4f}")
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
# 6. 5-Fold 交叉驗證
# ============================================================
print("\n\n【5-Fold 交叉驗證】")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    lgb.LGBMClassifier(**best_params),
    X_train_c, y_train_encoded, cv=cv, scoring='f1_macro'
)

print(f"F1 Macro scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================
# 7. 使用完整訓練集重新訓練並預測測試集
# ============================================================
print("\n\n【測試集預測】")
print("=" * 70)

# 使用完整訓練集訓練
final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X_train_c, y_train_encoded)

# 預測測試集
test_predictions = final_model.predict(X_test_c)
test_predictions_original = test_predictions + 1

# 輸出 Kaggle 格式
submission = pd.DataFrame({
    'Id': range(len(test_predictions)),
    'ASA_Rating': test_predictions_original
})

submission.to_csv('submission_tuned.csv', index=False)
print(f"預測結果已儲存至 submission_tuned.csv")

# 預測分布
print("\n測試集預測分布:")
pred_dist = pd.Series(test_predictions_original).value_counts().sort_index()
for asa, count in pred_dist.items():
    pct = count / len(test_predictions_original) * 100
    print(f"  ASA {asa}: {count} ({pct:.1f}%)")

# ============================================================
# 8. 與其他方法比較
# ============================================================
print("\n\n【方法比較】")
print("=" * 70)

print(f"{'方法':<25} {'Kaggle Score':>15} {'本地 F1':>12}")
print("-" * 55)
print(f"{'A: Baseline':<25} {'0.46513':>15} {'0.4517':>12}")
print(f"{'B: 完整特徵':<25} {'0.52588':>15} {'0.5122':>12}")
print(f"{'C: 交互特徵':<25} {'0.53120':>15} {'0.4974':>12}")
print(f"{'D: 融合(B+C)':<25} {'0.52592':>15} {'0.5125':>12}")
print(f"{'E: 超參數調優':<25} {'?':>15} {f1:>12.4f}")

print("\n" + "=" * 70)
print("方法 E: 超參數調優完成")
print("=" * 70)
