"""
產生模型評估報告：Classification Report 和 Confusion Matrix
使用方法 E 的最佳超參數
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang TC']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 載入並準備方法C的資料
# ============================================================
print("=" * 70)
print("產生模型評估報告")
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
print("\n產生交互特徵...")

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
# 3. 使用最佳超參數訓練模型
# ============================================================
print("\n使用方法 E 最佳超參數訓練模型...")

# 方法 E 的最佳超參數
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

# 切分驗證集
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_c, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)

print(f"訓練集: {X_tr.shape}")
print(f"驗證集: {X_val.shape}")

# 訓練模型
model = lgb.LGBMClassifier(**best_params)
model.fit(X_tr, y_tr)

# ============================================================
# 4. 驗證集評估
# ============================================================
print("\n" + "=" * 70)
print("驗證集評估結果")
print("=" * 70)

y_pred = model.predict(X_val)

f1 = f1_score(y_val, y_pred, average='macro')
acc = accuracy_score(y_val, y_pred)
kappa = cohen_kappa_score(y_val, y_pred)

print(f"\nF1 Macro: {f1:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Kappa:    {kappa:.4f}")

# Classification Report
print("\n" + "-" * 70)
print("Classification Report")
print("-" * 70)
report = classification_report(y_val, y_pred, target_names=['ASA 1', 'ASA 2', 'ASA 3', 'ASA 4'])
print(report)

# Confusion Matrix
print("-" * 70)
print("Confusion Matrix")
print("-" * 70)
cm = confusion_matrix(y_val, y_pred)
cm_df = pd.DataFrame(cm,
                     index=['真實 ASA 1', '真實 ASA 2', '真實 ASA 3', '真實 ASA 4'],
                     columns=['預測 ASA 1', '預測 ASA 2', '預測 ASA 3', '預測 ASA 4'])
print(cm_df)

# ============================================================
# 5. 5-Fold 交叉驗證
# ============================================================
print("\n" + "=" * 70)
print("5-Fold 交叉驗證")
print("=" * 70)

from sklearn.model_selection import cross_val_predict

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 使用 cross_val_predict 獲取所有樣本的預測
y_pred_cv = cross_val_predict(
    lgb.LGBMClassifier(**best_params),
    X_train_c, y_train_encoded, cv=cv, n_jobs=-1
)

f1_cv = f1_score(y_train_encoded, y_pred_cv, average='macro')
acc_cv = accuracy_score(y_train_encoded, y_pred_cv)
kappa_cv = cohen_kappa_score(y_train_encoded, y_pred_cv)

print(f"\n5-Fold CV 結果:")
print(f"F1 Macro: {f1_cv:.4f}")
print(f"Accuracy: {acc_cv:.4f}")
print(f"Kappa:    {kappa_cv:.4f}")

# 5-Fold CV Classification Report
print("\n" + "-" * 70)
print("5-Fold CV Classification Report")
print("-" * 70)
report_cv = classification_report(y_train_encoded, y_pred_cv, target_names=['ASA 1', 'ASA 2', 'ASA 3', 'ASA 4'])
print(report_cv)

# 5-Fold CV Confusion Matrix
print("-" * 70)
print("5-Fold CV Confusion Matrix")
print("-" * 70)
cm_cv = confusion_matrix(y_train_encoded, y_pred_cv)
cm_cv_df = pd.DataFrame(cm_cv,
                        index=['真實 ASA 1', '真實 ASA 2', '真實 ASA 3', '真實 ASA 4'],
                        columns=['預測 ASA 1', '預測 ASA 2', '預測 ASA 3', '預測 ASA 4'])
print(cm_cv_df)

# ============================================================
# 6. 各類別詳細分析
# ============================================================
print("\n" + "=" * 70)
print("各類別詳細分析 (5-Fold CV)")
print("=" * 70)

# 計算每個類別的錯誤分布
for i, asa_class in enumerate(['ASA 1', 'ASA 2', 'ASA 3', 'ASA 4']):
    true_count = (y_train_encoded == i).sum()
    correct = cm_cv[i, i]
    correct_rate = correct / true_count * 100

    print(f"\n{asa_class} (樣本數: {true_count})")
    print(f"  正確預測: {correct} ({correct_rate:.1f}%)")

    # 錯誤分布
    errors = []
    for j in range(4):
        if i != j:
            error_count = cm_cv[i, j]
            if error_count > 0:
                error_rate = error_count / true_count * 100
                errors.append(f"誤判為 ASA {j+1}: {error_count} ({error_rate:.1f}%)")

    if errors:
        print("  錯誤分布:")
        for e in errors:
            print(f"    - {e}")

# ============================================================
# 7. 產生 Confusion Matrix 視覺化圖表
# ============================================================
print("\n" + "=" * 70)
print("產生 Confusion Matrix 視覺化圖表")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 圖 1: 原始數值的 Confusion Matrix
ax1 = axes[0]
sns.heatmap(cm_cv, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['ASA 1', 'ASA 2', 'ASA 3', 'ASA 4'],
            yticklabels=['ASA 1', 'ASA 2', 'ASA 3', 'ASA 4'],
            annot_kws={'size': 12})
ax1.set_xlabel('預測類別', fontsize=12)
ax1.set_ylabel('真實類別', fontsize=12)
ax1.set_title('Confusion Matrix (5-Fold CV)\n原始數值', fontsize=14, fontweight='bold')

# 圖 2: 正規化的 Confusion Matrix (按行，顯示百分比)
cm_normalized = cm_cv.astype('float') / cm_cv.sum(axis=1)[:, np.newaxis] * 100
ax2 = axes[1]
sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Oranges', ax=ax2,
            xticklabels=['ASA 1', 'ASA 2', 'ASA 3', 'ASA 4'],
            yticklabels=['ASA 1', 'ASA 2', 'ASA 3', 'ASA 4'],
            annot_kws={'size': 12})
ax2.set_xlabel('預測類別', fontsize=12)
ax2.set_ylabel('真實類別', fontsize=12)
ax2.set_title('Confusion Matrix (5-Fold CV)\n正規化百分比 (%)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../專案紀錄/confusion_matrix.png', dpi=150, bbox_inches='tight', facecolor='white')
print("圖表已儲存至: 專案紀錄/confusion_matrix.png")

# 額外產生單一大圖（更詳細的版本）
fig2, ax = plt.subplots(figsize=(10, 8))

# 建立標註文字（同時顯示數量和百分比）
annot_text = np.empty_like(cm_cv, dtype=object)
for i in range(4):
    for j in range(4):
        count = cm_cv[i, j]
        pct = cm_normalized[i, j]
        annot_text[i, j] = f'{count}\n({pct:.1f}%)'

sns.heatmap(cm_cv, annot=annot_text, fmt='', cmap='YlOrRd', ax=ax,
            xticklabels=['ASA 1', 'ASA 2', 'ASA 3', 'ASA 4'],
            yticklabels=['ASA 1', 'ASA 2', 'ASA 3', 'ASA 4'],
            annot_kws={'size': 11},
            cbar_kws={'label': '樣本數'})
ax.set_xlabel('預測類別', fontsize=14)
ax.set_ylabel('真實類別', fontsize=14)
ax.set_title('ASA 麻醉風險分級預測 - Confusion Matrix\n方法 E: 超參數調優 (5-Fold CV)',
             fontsize=16, fontweight='bold', pad=20)

# 加入統計資訊
stats_text = f'F1 Macro: {f1_cv:.4f}  |  Accuracy: {acc_cv:.4f}  |  Kappa: {kappa_cv:.4f}'
fig2.text(0.5, 0.02, stats_text, ha='center', fontsize=12, style='italic')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('../專案紀錄/confusion_matrix_detailed.png', dpi=150, bbox_inches='tight', facecolor='white')
print("詳細圖表已儲存至: 專案紀錄/confusion_matrix_detailed.png")

plt.show()

print("\n" + "=" * 70)
print("評估報告產生完成")
print("=" * 70)
