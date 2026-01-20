# ASA 麻醉風險分級預測 (ASA Classification Prediction)

預測美國麻醉醫師學會 (American Society of Anesthesiologists, ASA) 麻醉風險分級的機器學習專案。

## 專案概述

| 項目 | 內容 |
|------|------|
| **目標** | 預測患者的 ASA 分級 (1-4 級) |
| **資料集** | [MOVER Dataset](https://mover.ics.uci.edu/index.html) (Kaggle 競賽) |
| **訓練集** | 14,734 筆 |
| **測試集** | 6,315 筆 |
| **評估指標** | F1 Macro Score |
| **最佳成績** | **0.53809** |

## 成績演進

| 方法 | 技術 | 特徵數 | Kaggle Score | 提升幅度 |
|------|------|--------|--------------|----------|
| A: Baseline | LightGBM | 13 | 0.46513 | - |
| B: 完整特徵 | LightGBM + 特徵工程 | 40 | 0.52588 | +13.1% |
| C: 交互特徵 | LightGBM + PolynomialFeatures | 70 | 0.53120 | +1.0% |
| D: 模型融合 | B+C 加權融合 | 110 | 0.52592 | -1.0% |
| **E: 超參數調優** | **Optuna 100 trials** | **70** | **0.53809** | **+1.3%** |
| F: SMOTE | 過採樣 | 70 | 0.51602 | -4.1% |
| G: 三階交互 | PolynomialFeatures (degree=3) | 90 | 0.53598 | -0.4% |

**整體提升: +15.7%** (0.46513 → 0.53809)

## 專案結構

```
ASA 麻醉風險分級預測/
├── 資料清洗/
│   ├── surgery_classification.py    # 手術名稱分類
│   └── medication_classification.py # 藥物分類
├── 方法A_Baseline/
│   └── baseline_model.py            # 基準模型
├── 方法B_完整特徵/
│   ├── feature_engineering.py       # 特徵工程
│   └── advanced_model.py            # 進階模型
├── 方法C_交互特徵/
│   └── interaction_features.py      # 交互特徵
├── 方法D_模型融合/
│   └── ensemble_model.py            # 模型融合
├── 方法E_超參數調優/
│   └── tuning_model.py              # Optuna 調優
├── 方法F_SMOTE/
│   └── improved_model_with_smote.py # SMOTE 過採樣
├── 方法G_三階交互特徵/
│   └── degree3_interaction_model.py # 三階交互特徵
├── 專案紀錄/
│   ├── 改善歷程總結.md              # 完整改善歷程
│   ├── 問題與解決方案.md            # 遇到的問題與解法
│   └── 成績追蹤.md                  # Kaggle 分數追蹤
└── README.md
```

## 技術亮點

### 1. 資料清洗
- **MNAR 分析**: 發現 HEIGHT 缺失與患者風險等級相關（低風險患者缺失率較高）
- **異常值處理**: HEIGHT < 100cm、WEIGHT = 0 轉為 NaN 後用分組中位數填補
- **文字標準化**: 處理空格、統一類別名稱

### 2. 特徵工程
基於文獻回顧，從原始資料提取 **27 個新特徵**：

| 類別 | 特徵數 | 範例 |
|------|--------|------|
| Lab 檢驗值 | 7 | lab_abnormal_total, lab_critical_HH |
| 用藥紀錄 | 7 | has_cardiac_med, has_diabetes_med |
| 導管使用 | 8 | catheter_count, has_cvc |
| 衍生特徵 | 5 | BMI, age_group, is_elderly |

### 3. 交互特徵
使用 `PolynomialFeatures` 產生 **30 個二階交互項**，捕捉特徵間的非線性關係。

### 4. 超參數調優
使用 **Optuna** 進行 100 次試驗，找到最佳參數組合：

```python
{
    'n_estimators': 401,
    'max_depth': 8,
    'learning_rate': 0.0424,
    'num_leaves': 61,
    'min_child_samples': 42,
    'subsample': 0.729,
    'colsample_bytree': 0.945,
    'reg_alpha': 0.035,
    'reg_lambda': 0.0002
}
```

## 類別分布

資料存在明顯的類別不平衡問題：

| ASA 分級 | 數量 | 比例 |
|----------|------|------|
| ASA 1 | 801 | 5.4% |
| ASA 2 | 5,044 | 34.2% |
| ASA 3 | 7,540 | 51.2% |
| ASA 4 | 1,349 | 9.2% |

**處理策略**: 使用 `class_weight='balanced'` 或 SMOTE 過採樣

## 環境需求

```bash
pip install pandas numpy scikit-learn lightgbm optuna imbalanced-learn
```

## 使用方式

```bash
# 1. 執行特徵工程
python 方法B_完整特徵/feature_engineering.py

# 2. 執行最佳模型 (方法E)
python 方法E_超參數調優/tuning_model.py

# 3. 產生 submission.csv 提交 Kaggle
```

## 關鍵學習

### 成功策略
- 系統性的資料清洗流程
- 文獻導向的特徵工程（BMI 是 ASA 分級的明確標準）
- 交互特徵捕捉非線性關係
- 超參數調優提升泛化能力

### 踩坑紀錄
1. **XGBoost 標籤錯誤**: 標籤需從 0 開始，非 1-4
2. **本地 vs Kaggle 分數不一致**: 本地驗證集分布可能與測試集不同
3. **模型融合反而下降**: 權重設定需根據實際測試集表現調整
4. **SMOTE 過採樣無效**: 合成樣本破壞原始分布，`class_weight='balanced'` 更好
5. **三階交互特徵無效**: 更高維度不一定更好，可能引入噪音

## 參考資料

- [ASA Physical Status Classification System](https://www.asahq.org/standards-and-guidelines/asa-physical-status-classification-system)
- BMI 與 ASA 分級關係文獻 (PubMed, Cochrane Library)

## 授權

本專案為 **北醫醫療 AI 實戰力養成班** 課程作業。

---

*最後更新: 2026-01-20*
*當前最佳成績: 方法 E (0.53809)*
