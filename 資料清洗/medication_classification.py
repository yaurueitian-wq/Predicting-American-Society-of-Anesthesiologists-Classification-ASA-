"""
藥物名稱標準化分類系統
基於 ATC (Anatomical Therapeutic Chemical) 分類系統和臨床麻醉用藥分類

功能：
1. 解析藥物資訊：藥物名稱、劑量、劑量單位、劑型、給藥途徑
2. 標準化藥物名稱：將不同寫法統一為標準名稱
3. 藥物分類：分類為臨床相關的藥物類別
4. 劑量標準化：統一劑量單位便於比較
"""

import re
import pandas as pd
from collections import defaultdict

# ============================================================
# 標準化藥物類別 (基於臨床麻醉用藥分類)
# ============================================================

DRUG_CATEGORIES = {
    'IV_FLUIDS': 'IV Fluids (輸液)',
    'CRYSTALLOIDS': 'Crystalloids (晶體液)',
    'COLLOIDS': 'Colloids (膠體液)',
    'SEDATIVES_HYPNOTICS': 'Sedatives/Hypnotics (鎮靜安眠)',
    'OPIOID_ANALGESICS': 'Opioid Analgesics (鴉片類止痛)',
    'NONOPIOID_ANALGESICS': 'Non-opioid Analgesics (非鴉片止痛)',
    'LOCAL_ANESTHETICS': 'Local Anesthetics (局部麻醉)',
    'NEUROMUSCULAR_BLOCKERS': 'Neuromuscular Blockers (神經肌肉阻斷)',
    'REVERSAL_AGENTS': 'Reversal Agents (拮抗劑)',
    'VASOPRESSORS_INOTROPES': 'Vasopressors/Inotropes (升壓/強心)',
    'ANTIHYPERTENSIVES': 'Antihypertensives (降壓藥)',
    'ANTIARRHYTHMICS': 'Antiarrhythmics (抗心律不整)',
    'ANTICOAGULANTS': 'Anticoagulants (抗凝血)',
    'ANTIBIOTICS': 'Antibiotics (抗生素)',
    'ANTIEMETICS': 'Antiemetics (止吐)',
    'STEROIDS': 'Steroids (類固醇)',
    'INSULIN': 'Insulin (胰島素)',
    'BRONCHODILATORS': 'Bronchodilators (支氣管擴張)',
    'ANTISEPTICS': 'Antiseptics (消毒劑)',
    'CONTRAST_AGENTS': 'Contrast Agents (顯影劑)',
    'GABAPENTINOIDS': 'Gabapentinoids (加巴噴丁類)',
    'GI_AGENTS': 'GI Agents (腸胃藥)',
    'IMMUNOSUPPRESSANTS': 'Immunosuppressants (免疫抑制)',
    'DIURETICS': 'Diuretics (利尿劑)',
    'ELECTROLYTES': 'Electrolytes (電解質)',
    'VITAMINS': 'Vitamins (維生素)',
    'ANTIHISTAMINES': 'Antihistamines (抗組織胺)',
    'OPHTHALMIC': 'Ophthalmic Agents (眼科用藥)',
    'OTHER': 'Other (其他)'
}

# ============================================================
# 藥物名稱標準化映射
# ============================================================

# 標準藥物名稱 -> 可能的變體寫法
DRUG_STANDARDIZATION = {
    # === 輸液 ===
    'Plasma-Lyte A': ['PLASMA-LYTE', 'PLASMALYTE'],
    'Lactated Ringers': ['LACTATED RINGER', 'LR ', 'RINGERS'],
    'Normal Saline': ['SODIUM CHLORIDE 0.9', 'NACL 0.9', 'NS '],
    'Dextrose 5%': ['DEXTROSE 5', 'D5W', 'D5 '],
    'Dextrose-Saline': ['DEXTROSE-NACL', 'D5NS', 'D5 0.9'],

    # === 鎮靜安眠 ===
    'Midazolam': ['MIDAZOLAM'],
    'Propofol': ['PROPOFOL'],
    'Ketamine': ['KETAMINE'],
    'Dexmedetomidine': ['DEXMEDETOMIDINE', 'PRECEDEX'],
    'Etomidate': ['ETOMIDATE'],
    'Lorazepam': ['LORAZEPAM', 'ATIVAN'],
    'Diazepam': ['DIAZEPAM', 'VALIUM'],

    # === 鴉片類止痛 ===
    'Fentanyl': ['FENTANYL', 'FENTANIL'],
    'Morphine': ['MORPHINE'],
    'Hydromorphone': ['HYDROMORPHONE', 'DILAUDID'],
    'Hydrocodone': ['HYDROCODONE'],
    'Oxycodone': ['OXYCODONE'],
    'Meperidine': ['MEPERIDINE', 'DEMEROL'],
    'Remifentanil': ['REMIFENTANIL', 'ULTIVA'],
    'Sufentanil': ['SUFENTANIL'],
    'Tramadol': ['TRAMADOL'],

    # === 非鴉片止痛 ===
    'Acetaminophen': ['ACETAMINOPHEN', 'PARACETAMOL', 'TYLENOL'],
    'Ibuprofen': ['IBUPROFEN', 'ADVIL', 'MOTRIN'],
    'Ketorolac': ['KETOROLAC', 'TORADOL'],
    'Celecoxib': ['CELECOXIB', 'CELEBREX'],
    'Meloxicam': ['MELOXICAM'],
    'Naproxen': ['NAPROXEN'],

    # === 局部麻醉 ===
    'Lidocaine': ['LIDOCAINE', 'XYLOCAINE'],
    'Bupivacaine': ['BUPIVACAINE', 'MARCAINE'],
    'Ropivacaine': ['ROPIVACAINE', 'NAROPIN'],
    'Tetracaine': ['TETRACAINE'],
    'Mepivacaine': ['MEPIVACAINE'],
    'Lidocaine-Epinephrine': ['LIDOCAINE-EPINEPHRINE', 'LIDOCAINE WITH EPI'],
    'Bupivacaine Liposome': ['BUPIVACAINE LIPOSOME', 'EXPAREL'],

    # === 神經肌肉阻斷劑 ===
    'Rocuronium': ['ROCURONIUM', 'ZEMURON'],
    'Succinylcholine': ['SUCCINYLCHOLINE', 'ANECTINE'],
    'Vecuronium': ['VECURONIUM'],
    'Cisatracurium': ['CISATRACURIUM', 'NIMBEX'],
    'Pancuronium': ['PANCURONIUM'],

    # === 拮抗劑 ===
    'Naloxone': ['NALOXONE', 'NARCAN'],
    'Flumazenil': ['FLUMAZENIL', 'ROMAZICON'],
    'Sugammadex': ['SUGAMMADEX', 'BRIDION'],
    'Neostigmine': ['NEOSTIGMINE'],
    'Glycopyrrolate': ['GLYCOPYRROLATE'],

    # === 升壓/強心 ===
    'Epinephrine': ['EPINEPHRINE', 'ADRENALINE'],
    'Norepinephrine': ['NOREPINEPHRINE', 'LEVOPHED'],
    'Phenylephrine': ['PHENYLEPHRINE', 'NEOSYNEPHRINE'],
    'Vasopressin': ['VASOPRESSIN'],
    'Dopamine': ['DOPAMINE'],
    'Dobutamine': ['DOBUTAMINE'],
    'Ephedrine': ['EPHEDRINE'],
    'Milrinone': ['MILRINONE'],

    # === 降壓藥 ===
    'Labetalol': ['LABETALOL'],
    'Esmolol': ['ESMOLOL', 'BREVIBLOC'],
    'Metoprolol': ['METOPROLOL', 'LOPRESSOR'],
    'Hydralazine': ['HYDRALAZINE'],
    'Nicardipine': ['NICARDIPINE', 'CARDENE'],
    'Nitroglycerin': ['NITROGLYCERIN', 'NTG'],
    'Nitroprusside': ['NITROPRUSSIDE', 'NIPRIDE'],
    'Amlodipine': ['AMLODIPINE'],
    'Lisinopril': ['LISINOPRIL'],

    # === 抗心律不整 ===
    'Amiodarone': ['AMIODARONE', 'CORDARONE'],
    'Adenosine': ['ADENOSINE'],
    'Diltiazem': ['DILTIAZEM', 'CARDIZEM'],
    'Verapamil': ['VERAPAMIL'],
    'Lidocaine (Cardiac)': ['LIDOCAINE.*CARDIAC'],
    'Atropine': ['ATROPINE'],

    # === 抗凝血 ===
    'Heparin': ['HEPARIN'],
    'Enoxaparin': ['ENOXAPARIN', 'LOVENOX'],
    'Warfarin': ['WARFARIN', 'COUMADIN'],
    'Aspirin': ['ASPIRIN', 'ASA '],
    'Clopidogrel': ['CLOPIDOGREL', 'PLAVIX'],

    # === 抗生素 ===
    'Cefazolin': ['CEFAZOLIN', 'ANCEF'],
    'Ceftriaxone': ['CEFTRIAXONE', 'ROCEPHIN'],
    'Cefepime': ['CEFEPIME', 'MAXIPIME'],
    'Cefoxitin': ['CEFOXITIN'],
    'Vancomycin': ['VANCOMYCIN'],
    'Piperacillin-Tazobactam': ['PIPERACILLIN-TAZOBACTAM', 'ZOSYN'],
    'Ampicillin': ['AMPICILLIN'],
    'Ampicillin-Sulbactam': ['AMPICILLIN-SULBACTAM', 'UNASYN'],
    'Metronidazole': ['METRONIDAZOLE', 'FLAGYL'],
    'Ciprofloxacin': ['CIPROFLOXACIN', 'CIPRO'],
    'Levofloxacin': ['LEVOFLOXACIN', 'LEVAQUIN'],
    'Azithromycin': ['AZITHROMYCIN', 'ZITHROMAX'],
    'Moxifloxacin': ['MOXIFLOXACIN'],
    'Gentamicin': ['GENTAMICIN'],
    'Clindamycin': ['CLINDAMYCIN'],

    # === 止吐 ===
    'Ondansetron': ['ONDANSETRON', 'ZOFRAN'],
    'Metoclopramide': ['METOCLOPRAMIDE', 'REGLAN'],
    'Promethazine': ['PROMETHAZINE', 'PHENERGAN'],
    'Scopolamine': ['SCOPOLAMINE'],
    'Dexamethasone (Antiemetic)': ['DEXAMETHASONE.*ANTIEMETIC'],
    'Droperidol': ['DROPERIDOL'],

    # === 類固醇 ===
    'Dexamethasone': ['DEXAMETHASONE'],
    'Methylprednisolone': ['METHYLPREDNISOLONE', 'SOLU-MEDROL'],
    'Hydrocortisone': ['HYDROCORTISONE'],
    'Prednisone': ['PREDNISONE'],
    'Prednisolone': ['PREDNISOLONE'],

    # === 胰島素 ===
    'Insulin Regular': ['INSULIN REGULAR', 'HUMULIN R', 'NOVOLIN R'],
    'Insulin Lispro': ['INSULIN LISPRO', 'HUMALOG'],
    'Insulin Aspart': ['INSULIN ASPART', 'NOVOLOG'],
    'Insulin Glargine': ['INSULIN GLARGINE', 'LANTUS'],

    # === 支氣管擴張 ===
    'Albuterol': ['ALBUTEROL', 'VENTOLIN', 'PROAIR'],
    'Ipratropium': ['IPRATROPIUM', 'ATROVENT'],

    # === 消毒劑 ===
    'Povidone-Iodine': ['POVIDONE-IODINE', 'BETADINE'],
    'Chlorhexidine': ['CHLORHEXIDINE', 'HIBICLENS'],

    # === 顯影劑 ===
    'Iopamidol': ['IOPAMIDOL', 'ISOVUE'],
    'Iohexol': ['IOHEXOL', 'OMNIPAQUE'],
    'Technetium': ['TECHNET'],

    # === 加巴噴丁類 ===
    'Gabapentin': ['GABAPENTIN', 'NEURONTIN'],
    'Pregabalin': ['PREGABALIN', 'LYRICA'],

    # === 腸胃藥 ===
    'Famotidine': ['FAMOTIDINE', 'PEPCID'],
    'Pantoprazole': ['PANTOPRAZOLE', 'PROTONIX'],
    'Omeprazole': ['OMEPRAZOLE', 'PRILOSEC'],
    'Docusate': ['DOCUSATE', 'COLACE'],
    'Senna': ['SENNA', 'SENOKOT'],
    'Naloxegol': ['NALOXEGOL', 'MOVANTIK'],
    'Bisacodyl': ['BISACODYL'],

    # === 免疫抑制 ===
    'Mycophenolate': ['MYCOPHENOLATE', 'CELLCEPT'],
    'Tacrolimus': ['TACROLIMUS', 'PROGRAF'],
    'Cyclosporine': ['CYCLOSPORINE'],

    # === 利尿劑 ===
    'Furosemide': ['FUROSEMIDE', 'LASIX'],
    'Mannitol': ['MANNITOL'],
    'Bumetanide': ['BUMETANIDE', 'BUMEX'],

    # === 電解質 ===
    'Potassium Chloride': ['POTASSIUM CHLORIDE', 'KCL'],
    'Calcium Gluconate': ['CALCIUM GLUCONATE'],
    'Calcium Chloride': ['CALCIUM CHLORIDE'],
    'Magnesium Sulfate': ['MAGNESIUM SULFATE'],
    'Sodium Bicarbonate': ['SODIUM BICARBONATE'],

    # === 抗組織胺 ===
    'Diphenhydramine': ['DIPHENHYDRAMINE', 'BENADRYL'],
    'Hydroxyzine': ['HYDROXYZINE', 'VISTARIL'],

    # === 眼科用藥 ===
    'Cyclopentolate': ['CYCLOPENTOLATE'],
    'Tropicamide': ['TROPICAMIDE'],
    'Phenylephrine Ophthalmic': ['PHENYLEPHRINE.*OP'],

    # === 其他常見 ===
    'Oxytocin': ['OXYTOCIN', 'PITOCIN'],
    'Carboprost': ['CARBOPROST', 'HEMABATE'],
    'Methergine': ['METHERGINE', 'METHYLERGONOVINE'],
}

# 藥物 -> 類別映射
DRUG_TO_CATEGORY = {
    # IV Fluids
    'Plasma-Lyte A': 'IV_FLUIDS',
    'Lactated Ringers': 'IV_FLUIDS',
    'Normal Saline': 'IV_FLUIDS',
    'Dextrose 5%': 'IV_FLUIDS',
    'Dextrose-Saline': 'IV_FLUIDS',

    # Sedatives/Hypnotics
    'Midazolam': 'SEDATIVES_HYPNOTICS',
    'Propofol': 'SEDATIVES_HYPNOTICS',
    'Ketamine': 'SEDATIVES_HYPNOTICS',
    'Dexmedetomidine': 'SEDATIVES_HYPNOTICS',
    'Etomidate': 'SEDATIVES_HYPNOTICS',
    'Lorazepam': 'SEDATIVES_HYPNOTICS',
    'Diazepam': 'SEDATIVES_HYPNOTICS',

    # Opioid Analgesics
    'Fentanyl': 'OPIOID_ANALGESICS',
    'Morphine': 'OPIOID_ANALGESICS',
    'Hydromorphone': 'OPIOID_ANALGESICS',
    'Hydrocodone': 'OPIOID_ANALGESICS',
    'Oxycodone': 'OPIOID_ANALGESICS',
    'Meperidine': 'OPIOID_ANALGESICS',
    'Remifentanil': 'OPIOID_ANALGESICS',
    'Sufentanil': 'OPIOID_ANALGESICS',
    'Tramadol': 'OPIOID_ANALGESICS',

    # Non-opioid Analgesics
    'Acetaminophen': 'NONOPIOID_ANALGESICS',
    'Ibuprofen': 'NONOPIOID_ANALGESICS',
    'Ketorolac': 'NONOPIOID_ANALGESICS',
    'Celecoxib': 'NONOPIOID_ANALGESICS',
    'Meloxicam': 'NONOPIOID_ANALGESICS',
    'Naproxen': 'NONOPIOID_ANALGESICS',

    # Local Anesthetics
    'Lidocaine': 'LOCAL_ANESTHETICS',
    'Bupivacaine': 'LOCAL_ANESTHETICS',
    'Ropivacaine': 'LOCAL_ANESTHETICS',
    'Tetracaine': 'LOCAL_ANESTHETICS',
    'Mepivacaine': 'LOCAL_ANESTHETICS',
    'Lidocaine-Epinephrine': 'LOCAL_ANESTHETICS',
    'Bupivacaine Liposome': 'LOCAL_ANESTHETICS',

    # Neuromuscular Blockers
    'Rocuronium': 'NEUROMUSCULAR_BLOCKERS',
    'Succinylcholine': 'NEUROMUSCULAR_BLOCKERS',
    'Vecuronium': 'NEUROMUSCULAR_BLOCKERS',
    'Cisatracurium': 'NEUROMUSCULAR_BLOCKERS',
    'Pancuronium': 'NEUROMUSCULAR_BLOCKERS',

    # Reversal Agents
    'Naloxone': 'REVERSAL_AGENTS',
    'Flumazenil': 'REVERSAL_AGENTS',
    'Sugammadex': 'REVERSAL_AGENTS',
    'Neostigmine': 'REVERSAL_AGENTS',
    'Glycopyrrolate': 'REVERSAL_AGENTS',

    # Vasopressors/Inotropes
    'Epinephrine': 'VASOPRESSORS_INOTROPES',
    'Norepinephrine': 'VASOPRESSORS_INOTROPES',
    'Phenylephrine': 'VASOPRESSORS_INOTROPES',
    'Vasopressin': 'VASOPRESSORS_INOTROPES',
    'Dopamine': 'VASOPRESSORS_INOTROPES',
    'Dobutamine': 'VASOPRESSORS_INOTROPES',
    'Ephedrine': 'VASOPRESSORS_INOTROPES',
    'Milrinone': 'VASOPRESSORS_INOTROPES',

    # Antihypertensives
    'Labetalol': 'ANTIHYPERTENSIVES',
    'Esmolol': 'ANTIHYPERTENSIVES',
    'Metoprolol': 'ANTIHYPERTENSIVES',
    'Hydralazine': 'ANTIHYPERTENSIVES',
    'Nicardipine': 'ANTIHYPERTENSIVES',
    'Nitroglycerin': 'ANTIHYPERTENSIVES',
    'Nitroprusside': 'ANTIHYPERTENSIVES',
    'Amlodipine': 'ANTIHYPERTENSIVES',
    'Lisinopril': 'ANTIHYPERTENSIVES',

    # Antiarrhythmics
    'Amiodarone': 'ANTIARRHYTHMICS',
    'Adenosine': 'ANTIARRHYTHMICS',
    'Diltiazem': 'ANTIARRHYTHMICS',
    'Verapamil': 'ANTIARRHYTHMICS',
    'Lidocaine (Cardiac)': 'ANTIARRHYTHMICS',
    'Atropine': 'ANTIARRHYTHMICS',

    # Anticoagulants
    'Heparin': 'ANTICOAGULANTS',
    'Enoxaparin': 'ANTICOAGULANTS',
    'Warfarin': 'ANTICOAGULANTS',
    'Aspirin': 'ANTICOAGULANTS',
    'Clopidogrel': 'ANTICOAGULANTS',

    # Antibiotics
    'Cefazolin': 'ANTIBIOTICS',
    'Ceftriaxone': 'ANTIBIOTICS',
    'Cefepime': 'ANTIBIOTICS',
    'Cefoxitin': 'ANTIBIOTICS',
    'Vancomycin': 'ANTIBIOTICS',
    'Piperacillin-Tazobactam': 'ANTIBIOTICS',
    'Ampicillin': 'ANTIBIOTICS',
    'Ampicillin-Sulbactam': 'ANTIBIOTICS',
    'Metronidazole': 'ANTIBIOTICS',
    'Ciprofloxacin': 'ANTIBIOTICS',
    'Levofloxacin': 'ANTIBIOTICS',
    'Azithromycin': 'ANTIBIOTICS',
    'Moxifloxacin': 'ANTIBIOTICS',
    'Gentamicin': 'ANTIBIOTICS',
    'Clindamycin': 'ANTIBIOTICS',

    # Antiemetics
    'Ondansetron': 'ANTIEMETICS',
    'Metoclopramide': 'ANTIEMETICS',
    'Promethazine': 'ANTIEMETICS',
    'Scopolamine': 'ANTIEMETICS',
    'Droperidol': 'ANTIEMETICS',

    # Steroids
    'Dexamethasone': 'STEROIDS',
    'Methylprednisolone': 'STEROIDS',
    'Hydrocortisone': 'STEROIDS',
    'Prednisone': 'STEROIDS',
    'Prednisolone': 'STEROIDS',

    # Insulin
    'Insulin Regular': 'INSULIN',
    'Insulin Lispro': 'INSULIN',
    'Insulin Aspart': 'INSULIN',
    'Insulin Glargine': 'INSULIN',

    # Bronchodilators
    'Albuterol': 'BRONCHODILATORS',
    'Ipratropium': 'BRONCHODILATORS',

    # Antiseptics
    'Povidone-Iodine': 'ANTISEPTICS',
    'Chlorhexidine': 'ANTISEPTICS',

    # Contrast Agents
    'Iopamidol': 'CONTRAST_AGENTS',
    'Iohexol': 'CONTRAST_AGENTS',
    'Technetium': 'CONTRAST_AGENTS',

    # Gabapentinoids
    'Gabapentin': 'GABAPENTINOIDS',
    'Pregabalin': 'GABAPENTINOIDS',

    # GI Agents
    'Famotidine': 'GI_AGENTS',
    'Pantoprazole': 'GI_AGENTS',
    'Omeprazole': 'GI_AGENTS',
    'Docusate': 'GI_AGENTS',
    'Senna': 'GI_AGENTS',
    'Naloxegol': 'GI_AGENTS',
    'Bisacodyl': 'GI_AGENTS',

    # Immunosuppressants
    'Mycophenolate': 'IMMUNOSUPPRESSANTS',
    'Tacrolimus': 'IMMUNOSUPPRESSANTS',
    'Cyclosporine': 'IMMUNOSUPPRESSANTS',

    # Diuretics
    'Furosemide': 'DIURETICS',
    'Mannitol': 'DIURETICS',
    'Bumetanide': 'DIURETICS',

    # Electrolytes
    'Potassium Chloride': 'ELECTROLYTES',
    'Calcium Gluconate': 'ELECTROLYTES',
    'Calcium Chloride': 'ELECTROLYTES',
    'Magnesium Sulfate': 'ELECTROLYTES',
    'Sodium Bicarbonate': 'ELECTROLYTES',

    # Antihistamines
    'Diphenhydramine': 'ANTIHISTAMINES',
    'Hydroxyzine': 'ANTIHISTAMINES',

    # Ophthalmic
    'Cyclopentolate': 'OPHTHALMIC',
    'Tropicamide': 'OPHTHALMIC',
    'Phenylephrine Ophthalmic': 'OPHTHALMIC',

    # Other
    'Oxytocin': 'OTHER',
    'Carboprost': 'OTHER',
    'Methergine': 'OTHER',
}

# ============================================================
# 劑量單位標準化
# ============================================================

DOSE_UNIT_STANDARDIZATION = {
    'MG': 'mg',
    'MCG': 'mcg',
    'GM': 'g',
    'G': 'g',
    'ML': 'mL',
    'L': 'L',
    'UNIT': 'units',
    'UNITS': 'units',
    'MEQ': 'mEq',
    '%': '%',
}

# ============================================================
# 給藥途徑標準化
# ============================================================

ROUTE_STANDARDIZATION = {
    'INTRAVENOUS': 'IV',
    'IntraVENOUS': 'IV',
    'IV': 'IV',
    'ORAL': 'PO',
    'Oral': 'PO',
    'PO': 'PO',
    'OR': 'PO',
    'SUBCUTANEOUS': 'SC',
    'Subcutaneous': 'SC',
    'SC': 'SC',
    'INTRAMUSCULAR': 'IM',
    'IntraMUSCULAR': 'IM',
    'IM': 'IM',
    'EPIDURAL': 'Epidural',
    'Epidural': 'Epidural',
    'PERINEURAL': 'Perineural',
    'PeriNEURAL': 'Perineural',
    'TOPICAL': 'Topical',
    'Topical': 'Topical',
    'INFILTRATION': 'Infiltration',
    'Infiltration': 'Infiltration',
    'INHALATION': 'Inhalation',
    'Inhalation': 'Inhalation',
    'TRANSDERMAL': 'Transdermal',
    'Transdermal': 'Transdermal',
    'INTRADERMAL': 'ID',
    'IntraDERMAL': 'ID',
    'IRRIGATION': 'Irrigation',
    'Irrigation': 'Irrigation',
    'INJECTION': 'Injection',
    'Injection': 'Injection',
    'RIGHT EYE': 'OD',
    'Right Eye': 'OD',
    'LEFT EYE': 'OS',
    'Left Eye': 'OS',
    'BOTH EYES': 'OU',
    'EACH NARIS': 'Nasal',
    'Each Naris': 'Nasal',
}


def parse_medication(med_str):
    """
    解析藥物資訊字串

    Parameters:
    -----------
    med_str : str
        原始藥物字串，格式為 'DRUG_NAME': 'ROUTE'

    Returns:
    --------
    dict : 包含解析結果
        - raw_drug: 原始藥物名稱
        - drug_standardized: 標準化藥物名稱
        - drug_category: 藥物分類代碼
        - drug_category_name: 藥物分類名稱
        - dose_value: 劑量數值
        - dose_unit: 劑量單位
        - dose_standardized: 標準化劑量字串
        - route_raw: 原始給藥途徑
        - route_standardized: 標準化給藥途徑
    """
    result = {
        'raw_drug': None,
        'drug_standardized': None,
        'drug_category': 'OTHER',
        'drug_category_name': DRUG_CATEGORIES['OTHER'],
        'dose_value': None,
        'dose_unit': None,
        'dose_standardized': None,
        'route_raw': None,
        'route_standardized': None
    }

    if pd.isna(med_str):
        return result

    # 解析格式: 'DRUG_NAME': 'ROUTE'
    match = re.match(r"'([^']+)':\s*'([^']+)'", str(med_str))
    if not match:
        return result

    raw_drug = match.group(1)
    raw_route = match.group(2)

    result['raw_drug'] = raw_drug
    result['route_raw'] = raw_route

    # 標準化給藥途徑
    result['route_standardized'] = ROUTE_STANDARDIZATION.get(raw_route, raw_route)

    # 標準化藥物名稱
    drug_upper = raw_drug.upper()
    standardized_drug = None

    for std_name, variants in DRUG_STANDARDIZATION.items():
        for variant in variants:
            if re.search(variant, drug_upper):
                standardized_drug = std_name
                break
        if standardized_drug:
            break

    if standardized_drug:
        result['drug_standardized'] = standardized_drug
        result['drug_category'] = DRUG_TO_CATEGORY.get(standardized_drug, 'OTHER')
        result['drug_category_name'] = DRUG_CATEGORIES.get(result['drug_category'], DRUG_CATEGORIES['OTHER'])
    else:
        # 嘗試從原始名稱提取藥物名
        name_match = re.match(r'^([A-Z][A-Z\s\-\(\)]+?)(?=\s*\d|\s*$)', drug_upper)
        if name_match:
            result['drug_standardized'] = name_match.group(1).strip().title()
        else:
            result['drug_standardized'] = raw_drug.split()[0].title() if raw_drug else None
        result['drug_category'] = 'OTHER'
        result['drug_category_name'] = DRUG_CATEGORIES['OTHER']

    # 提取劑量
    # 嘗試多種劑量格式
    dose_patterns = [
        r'(\d+(?:\.\d+)?)\s*(MG|MCG|GM|G|UNIT|UNITS|ML|L|MEQ)(?:/|$|\s)',
        r'(\d+(?:\.\d+)?)\s*(%)',
    ]

    for pattern in dose_patterns:
        dose_match = re.search(pattern, drug_upper)
        if dose_match:
            result['dose_value'] = float(dose_match.group(1))
            raw_unit = dose_match.group(2)
            result['dose_unit'] = DOSE_UNIT_STANDARDIZATION.get(raw_unit, raw_unit.lower())
            result['dose_standardized'] = f"{result['dose_value']} {result['dose_unit']}"
            break

    return result


def process_medication_column(df, med_col='Medication_Usage'):
    """
    處理整個 DataFrame 的藥物欄位

    Parameters:
    -----------
    df : DataFrame
        包含藥物欄位的 DataFrame
    med_col : str
        藥物欄位名

    Returns:
    --------
    DataFrame : 新增藥物相關欄位的 DataFrame
    """
    # 應用解析
    results = df[med_col].apply(parse_medication)

    # 展開結果到新欄位
    df['Drug_Standardized'] = results.apply(lambda x: x['drug_standardized'])
    df['Drug_Category'] = results.apply(lambda x: x['drug_category'])
    df['Drug_Category_Name'] = results.apply(lambda x: x['drug_category_name'])
    df['Dose_Value'] = results.apply(lambda x: x['dose_value'])
    df['Dose_Unit'] = results.apply(lambda x: x['dose_unit'])
    df['Dose_Standardized'] = results.apply(lambda x: x['dose_standardized'])
    df['Route_Standardized'] = results.apply(lambda x: x['route_standardized'])

    return df


if __name__ == '__main__':
    # 測試
    test_meds = [
        "'MIDAZOLAM HCL 2 MG/2ML IJ SOLN': 'IntraVENOUS'",
        "'FENTANYL CITRATE (PF) 100 MCG/2ML IJ SOLN': 'IntraVENOUS'",
        "'ACETAMINOPHEN 500 MG OR TABS': 'Oral'",
        "'HEPARIN SODIUM (PORCINE) 5000 UNIT/ML IJ SOLN': 'Subcutaneous'",
        "'CEFAZOLIN IV PUSH 1 GM IN SWFI 10 ML': 'IntraVENOUS'",
        "'PLASMA-LYTE A IV SOLN': 'IntraVENOUS'",
        "'ROPIVACAINE HCL 5 MG/ML IJ SOLN': 'PeriNEURAL'",
        "'INSULIN LISPRO (HUMAN) 100 UNIT/ML SC SOLN (UCI)': 'Subcutaneous'",
    ]

    print('=== 藥物解析測試 ===\n')
    for med in test_meds:
        result = parse_medication(med)
        print(f'原始: {med}')
        print(f'  標準化藥物: {result["drug_standardized"]}')
        print(f'  藥物分類: {result["drug_category_name"]}')
        print(f'  劑量: {result["dose_standardized"]}')
        print(f'  給藥途徑: {result["route_standardized"]}')
        print()
