"""
方法 K: 患者分群探索 (非監督式學習)

目標：
  對抽樣樣本患者進行分群，探索資料中自然存在的臨床表型

策略：
  1. 重用方法 B 的特徵工程（Lab、用藥、導管、衍生特徵）
  2. StandardScaler 標準化 + PCA 降維
  3. Elbow + Silhouette 法決定最佳 k
  4. K-Means 分群
  5. 分群結果對照 ASA_Rating 驗證
  6. 各群臨床特徵分析與視覺化
"""

import pandas as pd
import numpy as np
import re
import warnings
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    normalized_mutual_info_score, confusion_matrix
)

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================
# 0. 設定
# ============================================================
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(OUTPUT_DIR, '..', '資料清洗')
K_RANGE    = range(2, 8)   # 測試 k = 2 ~ 7

print("=" * 70)
print("方法 K: 患者分群探索 (PCA + K-Means)")
print("=" * 70)

# ============================================================
# 1. 載入資料
# ============================================================
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_cleaned.csv'))
print(f"\n訓練集: {train_df.shape}")
print(f"ASA 分布:\n{train_df['ASA_Rating'].value_counts().sort_index().to_string()}")

# ============================================================
# 2. 特徵工程（重用方法 B 邏輯）
# ============================================================
print("\n\n【特徵工程】")
print("-" * 70)

# --- Lab_Values ---
def extract_lab_features(lab_str):
    features = {
        'lab_count': 0, 'lab_abnormal_H': 0, 'lab_abnormal_L': 0,
        'lab_critical_HH': 0, 'lab_critical_LL': 0, 'lab_has_data': 0,
    }
    if pd.isna(lab_str):
        return features
    features['lab_has_data'] = 1
    matches = re.findall(r'\(([HLN]+)\)', str(lab_str))
    features['lab_count'] = len(matches)
    for s in matches:
        if s == 'HH':   features['lab_critical_HH'] += 1
        elif s == 'LL': features['lab_critical_LL'] += 1
        elif s == 'H':  features['lab_abnormal_H']  += 1
        elif s == 'L':  features['lab_abnormal_L']  += 1
    return features

# --- 特定 Lab 數值 ---
KEY_LABS = ['Creatinine', 'Hemoglobin', 'Glucose', 'Sodium',
            'Potassium', 'Platelets', 'Leukocytes^^corrected for nucleated erythrocytes']

def extract_key_lab_values(lab_str):
    features = {}
    for test in KEY_LABS:
        key = test.split(' ')[0].lower().replace('^', '')
        features[f'val_{key}'] = np.nan
    if pd.isna(lab_str):
        return features
    lab_str = str(lab_str)
    for test in KEY_LABS:
        key = test.split(' ')[0].lower().replace('^', '')
        pattern = rf'{re.escape(test)}:\s*([\d.]+)'
        m = re.search(pattern, lab_str, re.IGNORECASE)
        if m:
            features[f'val_{key}'] = float(m.group(1))
    return features

# --- Medication ---
def extract_med_features(row):
    features = {
        'med_count': 0, 'has_chronic_med': 0, 'has_cardiac_med': 0,
        'has_diabetes_med': 0, 'has_anticoagulant': 0, 'has_opioid': 0, 'has_sedative': 0,
    }
    med_str  = str(row.get('Medication_Usage', ''))
    drug_cat = str(row.get('Drug_Category', ''))
    drug_name = str(row.get('Drug_Standardized', '')).lower()

    if pd.notna(row.get('Medication_Usage')) and med_str != 'nan':
        features['med_count'] = len(med_str.split(','))
    chronic_cats = ['ANTIHYPERTENSIVES', 'CARDIAC', 'DIURETICS', 'ANTIARRHYTHMICS']
    if drug_cat in chronic_cats:
        features['has_chronic_med'] = 1
        features['has_cardiac_med'] = 1
    if drug_cat == 'INSULIN' or 'metformin' in drug_name or 'glipizide' in drug_name:
        features['has_diabetes_med'] = 1
        features['has_chronic_med']  = 1
    if drug_cat == 'ANTICOAGULANTS':     features['has_anticoagulant'] = 1
    if drug_cat == 'OPIOID_ANALGESICS':  features['has_opioid']        = 1
    if drug_cat == 'SEDATIVES_HYPNOTICS': features['has_sedative']     = 1
    return features

# --- Catheter ---
def extract_catheter_features(cath_str):
    features = {
        'catheter_count': 0, 'has_catheter': 0, 'has_piv': 0,
        'has_urinary': 0, 'has_cvc': 0, 'has_arterial': 0,
        'has_chest_tube': 0, 'has_wound': 0,
    }
    if pd.isna(cath_str):
        return features
    cath_str = str(cath_str).upper()
    features['has_catheter']    = 1
    features['catheter_count']  = len(cath_str.split(','))
    if 'PIV' in cath_str or 'PERIPHERAL IV' in cath_str: features['has_piv']        = 1
    if 'URINARY' in cath_str:                             features['has_urinary']    = 1
    if any(t in cath_str for t in ['CVC', 'CENTRAL', 'PICC']): features['has_cvc']  = 1
    if 'ARTERIAL' in cath_str or 'ART' in cath_str:      features['has_arterial']   = 1
    if 'CHEST TUBE' in cath_str:                          features['has_chest_tube'] = 1
    if any(t in cath_str for t in ['WOUND', 'INCISION', 'DRAIN']): features['has_wound'] = 1
    return features

# --- 衍生特徵 ---
def extract_derived_features(row):
    age = row.get('Age', 0)
    bmi = row.get('BMI', 0) or 0
    return {
        'age_group':      0 if age < 40 else (1 if age < 60 else (2 if age < 75 else 3)),
        'is_elderly':     int(age >= 65),
        'is_obese':       int(bmi >= 30),
        'is_morbid_obese': int(bmi >= 40),
    }

print("  提取 Lab 特徵...")
lab_feats  = train_df['Lab_Values'].apply(extract_lab_features).apply(pd.Series)
lab_feats['lab_abnormal_total'] = (
    lab_feats['lab_abnormal_H'] + lab_feats['lab_abnormal_L'] +
    lab_feats['lab_critical_HH'] + lab_feats['lab_critical_LL']
)

print("  提取 Lab 數值特徵...")
key_lab_vals = train_df['Lab_Values'].apply(extract_key_lab_values).apply(pd.Series)

print("  提取 Medication 特徵...")
med_feats  = train_df.apply(extract_med_features, axis=1).apply(pd.Series)

print("  提取 Catheter 特徵...")
cath_feats = train_df['Catheter_Use'].apply(extract_catheter_features).apply(pd.Series)

print("  提取衍生特徵...")
other_feats = train_df.apply(extract_derived_features, axis=1).apply(pd.Series)

# ============================================================
# 3. 合併特徵、前處理
# ============================================================
print("\n【前處理】")
print("-" * 70)

# 基礎數值特徵
base_numeric = ['Age', 'WEIGHT', 'BMI', 'Surgery_Count']
base_numeric = [c for c in base_numeric if c in train_df.columns]

# 類別特徵（Label Encoding）
base_cat = ['Gender', 'ICU_Patient', 'Anesthesia_Method',
            'Patient_Source', 'Surgery_Category']
base_cat = [c for c in base_cat if c in train_df.columns]

cat_encoded = train_df[base_cat].copy()
for col in base_cat:
    le = LabelEncoder()
    cat_encoded[col] = le.fit_transform(cat_encoded[col].fillna('Unknown').astype(str))

# 合併
X = pd.concat([
    train_df[base_numeric].reset_index(drop=True),
    cat_encoded.reset_index(drop=True),
    lab_feats.reset_index(drop=True),
    key_lab_vals.reset_index(drop=True),
    med_feats.reset_index(drop=True),
    cath_feats.reset_index(drop=True),
    other_feats.reset_index(drop=True),
], axis=1)

y_asa = train_df['ASA_Rating'].values

# 缺失值填補（Lab 數值用中位數，其餘用 0）
for col in X.columns:
    if col.startswith('val_'):
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna(0)

print(f"  特徵總數: {X.shape[1]}")
print(f"  樣本數  : {X.shape[0]}")

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 4. PCA 降維
# ============================================================
print("\n【PCA 降維】")
print("-" * 70)

pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)

# 累積解釋變異
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components_90 = np.searchsorted(cum_var, 0.90) + 1
n_components_95 = np.searchsorted(cum_var, 0.95) + 1

print(f"  PC 數達 90% 解釋變異: {n_components_90}")
print(f"  PC 數達 95% 解釋變異: {n_components_95}")
print(f"  前 5 PC 解釋變異: {pca_full.explained_variance_ratio_[:5].round(3)}")

# 選擇 90% 解釋變異的 PC 數做分群，保留 2D 做視覺化
N_PCA = n_components_90
pca_cluster = PCA(n_components=N_PCA, random_state=42)
X_pca = pca_cluster.fit_transform(X_scaled)

pca_2d = PCA(n_components=2, random_state=42)
X_2d   = pca_2d.fit_transform(X_scaled)

# PCA 解釋變異圖
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(cum_var[:30]) + 1), cum_var[:30], 'b-o', markersize=4)
ax.axhline(0.90, color='red',   linestyle='--', label='90%', alpha=0.7)
ax.axhline(0.95, color='orange', linestyle='--', label='95%', alpha=0.7)
ax.axvline(n_components_90, color='red',   linestyle=':', alpha=0.5)
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('Cumulative Explained Variance')
ax.set_title('PCA Cumulative Explained Variance')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_explained_variance.png'), dpi=150)
plt.close()
print(f"\n  已儲存 pca_explained_variance.png")

# ============================================================
# 5. 最佳 k 選擇 (Elbow + Silhouette)
# ============================================================
print("\n【尋找最佳分群數 k】")
print("-" * 70)
print(f"  測試範圍: k = {list(K_RANGE)}")

inertias    = []
silhouettes = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_pca, labels, sample_size=3000, random_state=42)
    silhouettes.append(sil)
    print(f"  k={k}: inertia={km.inertia_:.0f}  silhouette={sil:.4f}")

# 繪製 Elbow + Silhouette
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(list(K_RANGE), inertias, 'b-o')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia (Within-cluster SSE)')
axes[0].set_title('Elbow Method')
axes[0].grid(alpha=0.3)

axes[1].plot(list(K_RANGE), silhouettes, 'g-o')
best_k_idx = int(np.argmax(silhouettes))
best_k     = list(K_RANGE)[best_k_idx]
axes[1].axvline(best_k, color='red', linestyle='--', label=f'Best k={best_k}')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score by k')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'elbow_silhouette.png'), dpi=150)
plt.close()
print(f"\n  最佳 k = {best_k} (Silhouette = {silhouettes[best_k_idx]:.4f})")
print(f"  已儲存 elbow_silhouette.png")

# ============================================================
# 6. 最終分群
# ============================================================
print(f"\n【K-Means 分群 (k={best_k})】")
print("-" * 70)

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
cluster_labels = km_final.fit_predict(X_pca)

# 驗證指標
ari = adjusted_rand_score(y_asa, cluster_labels)
nmi = normalized_mutual_info_score(y_asa, cluster_labels)
sil = silhouette_score(X_pca, cluster_labels, sample_size=3000, random_state=42)

print(f"  Silhouette Score         : {sil:.4f}")
print(f"  Adjusted Rand Index (ARI): {ari:.4f}  (對照 ASA_Rating)")
print(f"  NMI                      : {nmi:.4f}  (對照 ASA_Rating)")

# 加回 DataFrame
result_df = train_df.copy()
result_df['Cluster'] = cluster_labels

# ============================================================
# 7. 分群統計分析
# ============================================================
print(f"\n【各群臨床特徵摘要】")
print("=" * 70)

# 加入工程特徵到結果
result_df_feat = result_df.copy()
for col in ['lab_abnormal_total', 'lab_count', 'lab_has_data']:
    result_df_feat[col] = lab_feats[col].values
for col in ['med_count', 'has_chronic_med', 'has_cardiac_med', 'has_diabetes_med']:
    result_df_feat[col] = med_feats[col].values
for col in ['catheter_count', 'has_catheter', 'has_cvc', 'has_urinary']:
    result_df_feat[col] = cath_feats[col].values
result_df_feat['is_elderly'] = other_feats['is_elderly'].values
result_df_feat['is_obese']   = other_feats['is_obese'].values

summary_cols = ['Age', 'BMI', 'lab_abnormal_total', 'lab_count',
                'med_count', 'catheter_count',
                'has_chronic_med', 'has_cardiac_med', 'has_diabetes_med',
                'has_cvc', 'has_urinary', 'is_elderly', 'is_obese',
                'ICU_Patient']

cluster_summary = {}
for k in range(best_k):
    mask = result_df_feat['Cluster'] == k
    sub  = result_df_feat[mask]
    n    = mask.sum()
    row  = {'N': n, 'Pct': f"{n/len(result_df_feat)*100:.1f}%"}

    # 數值型：中位數
    for col in ['Age', 'BMI', 'lab_abnormal_total', 'lab_count', 'med_count', 'catheter_count']:
        if col in sub.columns:
            row[col] = f"{sub[col].median():.1f}"

    # 二元型：百分比
    for col in ['has_chronic_med', 'has_cardiac_med', 'has_diabetes_med',
                'has_cvc', 'has_urinary', 'is_elderly', 'is_obese']:
        if col in sub.columns:
            row[col] = f"{sub[col].mean()*100:.1f}%"

    # ICU_Patient
    if 'ICU_Patient' in sub.columns:
        icu_vals = sub['ICU_Patient'].astype(str).str.lower()
        row['ICU_Patient'] = f"{(icu_vals == 'yes').mean()*100:.1f}%"

    # ASA 分布
    asa_dist = sub['ASA_Rating'].value_counts(normalize=True).sort_index()
    row['ASA_dist'] = ' | '.join([f"ASA{i}:{v*100:.0f}%" for i, v in asa_dist.items()])
    row['主要ASA']  = sub['ASA_Rating'].value_counts().idxmax()

    cluster_summary[f'Cluster {k}'] = row
    print(f"\n  Cluster {k} (n={n}, {n/len(result_df_feat)*100:.1f}%):")
    print(f"    年齡中位數        : {row.get('Age', 'N/A')}")
    print(f"    BMI 中位數        : {row.get('BMI', 'N/A')}")
    print(f"    Lab 異常數中位數  : {row.get('lab_abnormal_total', 'N/A')}")
    print(f"    用藥數中位數      : {row.get('med_count', 'N/A')}")
    print(f"    導管數中位數      : {row.get('catheter_count', 'N/A')}")
    print(f"    慢性病用藥比例    : {row.get('has_chronic_med', 'N/A')}")
    print(f"    糖尿病用藥比例    : {row.get('has_diabetes_med', 'N/A')}")
    print(f"    老年人 (≥65) 比例 : {row.get('is_elderly', 'N/A')}")
    print(f"    ICU 患者比例      : {row.get('ICU_Patient', 'N/A')}")
    print(f"    ASA 分布          : {row.get('ASA_dist', 'N/A')}")

# ============================================================
# 8. 視覺化
# ============================================================
print("\n【產生視覺化圖表】")
print("-" * 70)

CLUSTER_COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800',
                  '#9C27B0', '#00BCD4', '#795548']
ASA_COLORS     = {1: '#81C784', 2: '#FFD54F', 3: '#FF8A65', 4: '#E53935'}

# --- 圖 1: PCA 散點圖（分群 vs ASA） ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左：按分群著色
for k in range(best_k):
    mask = cluster_labels == k
    axes[0].scatter(
        X_2d[mask, 0], X_2d[mask, 1],
        c=CLUSTER_COLORS[k], label=f'Cluster {k} (n={mask.sum()})',
        alpha=0.4, s=8, edgecolors='none'
    )
axes[0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].set_title(f'PCA — Colored by Cluster (k={best_k})')
axes[0].legend(markerscale=3, fontsize=8)
axes[0].grid(alpha=0.2)

# 右：按 ASA_Rating 著色
for asa_val in sorted(np.unique(y_asa)):
    mask = y_asa == asa_val
    axes[1].scatter(
        X_2d[mask, 0], X_2d[mask, 1],
        c=ASA_COLORS.get(asa_val, '#999999'),
        label=f'ASA {asa_val} (n={mask.sum()})',
        alpha=0.4, s=8, edgecolors='none'
    )
axes[1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].set_title('PCA — Colored by ASA Rating')
axes[1].legend(markerscale=3, fontsize=8)
axes[1].grid(alpha=0.2)

plt.suptitle('Patient Clustering — PCA Projection', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_scatter.png'), dpi=150)
plt.close()
print("  已儲存 pca_scatter.png")

# --- 圖 2: Cluster vs ASA 交叉熱圖 ---
cross_tab = pd.crosstab(
    result_df['Cluster'], result_df['ASA_Rating'],
    rownames=['Cluster'], colnames=['ASA Rating'],
    normalize='index'
) * 100

fig, ax = plt.subplots(figsize=(7, max(3, best_k)))
sns.heatmap(
    cross_tab, annot=True, fmt='.1f', cmap='YlOrRd',
    linewidths=0.5, ax=ax, cbar_kws={'label': 'Row %'}
)
ax.set_title('Cluster vs ASA Rating Distribution (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('ASA Rating')
ax.set_ylabel('Cluster')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'cluster_asa_heatmap.png'), dpi=150)
plt.close()
print("  已儲存 cluster_asa_heatmap.png")

# --- 圖 3: 各群關鍵數值特徵 Boxplot ---
numeric_plot_cols = ['Age', 'BMI', 'lab_abnormal_total', 'med_count', 'catheter_count']
numeric_plot_cols = [c for c in numeric_plot_cols if c in result_df_feat.columns]

fig, axes = plt.subplots(1, len(numeric_plot_cols), figsize=(4 * len(numeric_plot_cols), 5))
if len(numeric_plot_cols) == 1:
    axes = [axes]

for ax, col in zip(axes, numeric_plot_cols):
    data_by_cluster = [
        result_df_feat[result_df_feat['Cluster'] == k][col].dropna().values
        for k in range(best_k)
    ]
    bp = ax.boxplot(data_by_cluster, patch_artist=True, notch=False)
    for patch, color in zip(bp['boxes'], CLUSTER_COLORS[:best_k]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels([f'C{k}' for k in range(best_k)])
    ax.set_title(col.replace('_', '\n'), fontsize=9)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('Key Feature Distribution by Cluster', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'cluster_feature_boxplot.png'), dpi=150)
plt.close()
print("  已儲存 cluster_feature_boxplot.png")

# --- 圖 4: 各群 ASA 堆疊長條圖 ---
cross_tab_count = pd.crosstab(result_df['Cluster'], result_df['ASA_Rating'])
cross_tab_pct   = cross_tab_count.div(cross_tab_count.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 絕對數量
cross_tab_count.plot(
    kind='bar', ax=axes[0], stacked=True,
    color=[ASA_COLORS.get(c, '#999') for c in cross_tab_count.columns],
    alpha=0.85, edgecolor='white'
)
axes[0].set_title('ASA Count per Cluster', fontweight='bold')
axes[0].set_xlabel('Cluster')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(title='ASA', bbox_to_anchor=(1.02, 1), loc='upper left')

# 百分比
cross_tab_pct.plot(
    kind='bar', ax=axes[1], stacked=True,
    color=[ASA_COLORS.get(c, '#999') for c in cross_tab_pct.columns],
    alpha=0.85, edgecolor='white'
)
axes[1].set_title('ASA Distribution per Cluster (%)', fontweight='bold')
axes[1].set_xlabel('Cluster')
axes[1].set_ylabel('Percentage (%)')
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(title='ASA', bbox_to_anchor=(1.02, 1), loc='upper left')

plt.suptitle('ASA Rating Distribution Across Clusters', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'cluster_asa_bar.png'), dpi=150)
plt.close()
print("  已儲存 cluster_asa_bar.png")

# ============================================================
# 9. 儲存結果
# ============================================================
print("\n【儲存分群結果】")
print("-" * 70)

# 加入工程特徵一起輸出
output_cols = ['Gender', 'ICU_Patient', 'Age', 'WEIGHT', 'BMI',
               'Anesthesia_Method', 'Patient_Source', 'Surgery_Category',
               'ASA_Rating', 'Cluster']
output_cols = [c for c in output_cols if c in result_df.columns]

result_out = result_df[output_cols].copy()
for col in ['lab_abnormal_total', 'lab_count', 'lab_has_data',
            'med_count', 'has_chronic_med', 'has_cardiac_med', 'has_diabetes_med',
            'catheter_count', 'has_catheter', 'has_cvc',
            'is_elderly', 'is_obese']:
    if col in result_df_feat.columns:
        result_out[col] = result_df_feat[col].values

result_out.to_csv(os.path.join(OUTPUT_DIR, 'clustering_results.csv'), index=False)
print(f"  已儲存 clustering_results.csv ({result_out.shape})")

# ============================================================
# 10. 最終摘要
# ============================================================
print("\n\n" + "=" * 70)
print("方法 K: 患者分群探索 — 結果摘要")
print("=" * 70)
print(f"\n  分群數 k             : {best_k}")
print(f"  使用 PC 數           : {N_PCA} ({pca_cluster.explained_variance_ratio_.sum()*100:.1f}% 變異)")
print(f"  Silhouette Score    : {sil:.4f}")
print(f"  ARI (vs ASA_Rating) : {ari:.4f}")
print(f"  NMI (vs ASA_Rating) : {nmi:.4f}")

print(f"\n  ARI 解讀:")
if ari > 0.3:
    print(f"    -> 分群與 ASA 分級有顯著對應關係")
elif ari > 0.1:
    print(f"    -> 分群與 ASA 分級有部分對應，資料中存在 ASA 之外的結構")
else:
    print(f"    -> 分群捕捉到的是 ASA 以外的臨床維度（患者表型多樣性）")

print(f"\n  產生檔案:")
print(f"    pca_explained_variance.png  — PCA 解釋變異曲線")
print(f"    elbow_silhouette.png        — Elbow + Silhouette 選 k")
print(f"    pca_scatter.png             — PCA 散點（分群 vs ASA）")
print(f"    cluster_asa_heatmap.png     — Cluster × ASA 交叉熱圖")
print(f"    cluster_feature_boxplot.png — 各群特徵分布 Boxplot")
print(f"    cluster_asa_bar.png         — 各群 ASA 堆疊長條圖")
print(f"    clustering_results.csv      — 含分群標籤的資料集")
print("\n" + "=" * 70)
