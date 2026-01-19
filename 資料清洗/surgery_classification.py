"""
手術名稱標準化分類系統
基於 CPT (Current Procedural Terminology) 和 ICHI (International Classification of Health Interventions)

CPT 手術分類體系：
- Integumentary System (皮膚系統): 10030-19499
- Musculoskeletal System (肌肉骨骼系統): 20100-29999
- Respiratory System (呼吸系統): 30000-32999
- Cardiovascular System (心血管系統): 33016-37799
- Hemic and Lymphatic Systems (血液淋巴系統): 38100-38999
- Digestive System (消化系統): 40490-49999
- Urinary System (泌尿系統): 50010-53899
- Male Genital System (男性生殖系統): 54000-55899
- Female Genital System (女性生殖系統): 56405-58999
- Endocrine System (內分泌系統): 60000-60699
- Nervous System (神經系統): 61000-64999
- Eye and Ocular Adnexa (眼科): 65091-68899
- Auditory System (聽覺系統): 69000-69979
"""

import re
import pandas as pd

# ============================================================
# 標準化手術類別 (基於 CPT/ICHI)
# ============================================================

SURGERY_CATEGORIES = {
    # 主要手術系統分類
    'INTEGUMENTARY': 'Integumentary System',           # 皮膚、軟組織
    'MUSCULOSKELETAL': 'Musculoskeletal System',       # 骨骼、關節、肌肉
    'RESPIRATORY': 'Respiratory System',               # 呼吸系統
    'CARDIOVASCULAR': 'Cardiovascular System',         # 心血管
    'HEMIC_LYMPHATIC': 'Hemic and Lymphatic Systems',  # 血液淋巴
    'DIGESTIVE': 'Digestive System',                   # 消化系統
    'URINARY': 'Urinary System',                       # 泌尿系統
    'MALE_GENITAL': 'Male Genital System',             # 男性生殖
    'FEMALE_GENITAL': 'Female Genital System',         # 女性生殖
    'ENDOCRINE': 'Endocrine System',                   # 內分泌
    'NERVOUS': 'Nervous System',                       # 神經系統
    'EYE': 'Eye and Ocular Adnexa',                    # 眼科
    'AUDITORY': 'Auditory System',                     # 聽覺
    'INTERVENTIONAL': 'Interventional Radiology',      # 介入性放射
    'TRANSPLANT': 'Transplant Surgery',                # 器官移植
    'OTHER': 'Other/Unclassified'                      # 其他
}

# ============================================================
# 關鍵字映射規則
# ============================================================

# 解剖部位關鍵字
ANATOMY_KEYWORDS = {
    # 肌肉骨骼系統
    'MUSCULOSKELETAL': [
        'SPINE', 'VERTEBR', 'LUMBAR', 'CERVICAL', 'THORACIC', 'SACRAL',
        'FEMUR', 'TIBIA', 'FIBULA', 'HUMERUS', 'RADIUS', 'ULNA', 'CLAVICLE',
        'PELVIS', 'HIP', 'KNEE', 'ANKLE', 'SHOULDER', 'ELBOW', 'WRIST',
        'FINGER', 'TOE', 'FOOT', 'HAND', 'FRACTURE', 'BONE', 'JOINT',
        'ARTHRO', 'ORTHO', 'TENDON', 'LIGAMENT', 'MENISCUS', 'CARTILAGE',
        'ROTATOR CUFF', 'ACL', 'EXTREMITY', 'ORIF', 'LAMINECTOMY', 'DISCECTOMY',
        'FUSION', 'ARTHROPLASTY', 'ARTHROSCOPY', 'AMPUTATION'
    ],

    # 消化系統
    'DIGESTIVE': [
        'STOMACH', 'GASTRIC', 'GASTR', 'INTESTIN', 'COLON', 'COLECT',
        'RECTUM', 'RECTAL', 'ANUS', 'ANAL', 'LIVER', 'HEPAT', 'BILIARY',
        'GALLBLADDER', 'CHOLECYST', 'PANCREA', 'SPLEEN', 'SPLENIC',
        'APPENDIX', 'APPENDECT', 'HERNIA', 'ESOPHAG', 'DUODEN',
        'JEJUN', 'ILEUM', 'ILEOST', 'COLOST', 'BOWEL', 'GI ', 'EGD',
        'LAPAROTOMY', 'ABDOMEN', 'ABDOMINAL', 'PERITON', 'OMENT'
    ],

    # 心血管系統
    'CARDIOVASCULAR': [
        'HEART', 'CARDIAC', 'CORONARY', 'CABG', 'BYPASS', 'VALVE',
        'AORT', 'ARTERY', 'ARTERIAL', 'VEIN', 'VENOUS', 'VASCULAR',
        'ANEURYSM', 'EMBOL', 'THROMB', 'ENDARTERECT', 'PACEMAKER',
        'DEFIBRILLATOR', 'ICD', 'STENT', 'ANGIOPLASTY', 'CATH',
        'AV FISTULA', 'DIALYSIS ACCESS', 'VARICOSE'
    ],

    # 呼吸系統
    'RESPIRATORY': [
        'LUNG', 'PULMONARY', 'BRONCH', 'TRACHEA', 'TRACHEOSTOMY',
        'THORAC', 'CHEST', 'PLEURA', 'LOBECTOMY', 'PNEUMONECTOMY',
        'MEDIASTIN', 'VATS', 'THORACOSCOP'
    ],

    # 泌尿系統
    'URINARY': [
        'KIDNEY', 'RENAL', 'NEPHR', 'URETER', 'BLADDER', 'CYSTOSCOPY',
        'URETHRA', 'URINARY', 'LITHOTRIPSY', 'NEPHROSTOMY', 'TURBT',
        'PYELOPLASTY'
    ],

    # 男性生殖系統
    'MALE_GENITAL': [
        'PROSTAT', 'TURP', 'TESTIC', 'SCROTUM', 'SCROTAL', 'ORCHIE',
        'PENIS', 'PENILE', 'CIRCUMCISION', 'HYDROCELE', 'VARICOCELE',
        'VASECTOMY', 'EPIDIDYM'
    ],

    # 女性生殖系統
    'FEMALE_GENITAL': [
        'UTERUS', 'UTERINE', 'HYSTERECT', 'OVARY', 'OVARIAN', 'OOPHOR',
        'FALLOPIAN', 'SALPINGO', 'CERVIX', 'CERVICAL', 'VAGINA', 'VAGINAL',
        'VULVA', 'VULVAR', 'LABIA', 'ENDOMETRI', 'CURETTAGE', 'D&C',
        'CESAREAN', 'C-SECTION', 'MYOMECT', 'BSO'
    ],

    # 神經系統
    'NERVOUS': [
        'BRAIN', 'CEREBR', 'CRANIO', 'CRANIECT', 'INTRACRANIAL',
        'NERVE', 'NEURAL', 'NEURO', 'SPINAL CORD', 'SHUNT', 'VP SHUNT',
        'DEEP BRAIN', 'TUMOR.*BRAIN', 'SUBDURAL', 'EPIDURAL.*HEMATOMA',
        'ANEURYSM.*CEREBR', 'CAROTID'
    ],

    # 內分泌系統
    'ENDOCRINE': [
        'THYROID', 'PARATHYROID', 'ADRENAL', 'PITUITARY', 'THYROIDECTOMY'
    ],

    # 眼科
    'EYE': [
        'EYE', 'OCULAR', 'OPHTHALM', 'RETINA', 'CATARACT', 'CORNEA',
        'VITRECT', 'GLAUCOMA', 'BLEPHAR', 'ORBIT', 'LACRIM', 'EYELID',
        'LENS', 'PTERYGIUM', 'STRABISMUS'
    ],

    # 聽覺系統
    'AUDITORY': [
        'EAR', 'COCHLEAR', 'MASTOID', 'TYMPAN', 'MYRINGOTOMY',
        'STAPEDECTOMY', 'OSSICULAR'
    ],

    # 皮膚系統
    'INTEGUMENTARY': [
        'SKIN', 'DERMIS', 'SUBCUTANEOUS', 'WOUND', 'DEBRIDEMENT',
        'GRAFT', 'FLAP', 'LESION', 'EXCISION.*SKIN', 'LIPOMA',
        'ABSCESS', 'INCISION AND DRAINAGE', 'BURN', 'SCAR',
        'BREAST', 'MASTECT', 'MAMMOPLASTY', 'AUGMENTATION.*BREAST',
        'BIOPSY.*SKIN', 'MOHS'
    ],

    # 血液淋巴系統
    'HEMIC_LYMPHATIC': [
        'LYMPH', 'LYMPHADENECT', 'SENTINEL', 'SPLENECTOMY',
        'BONE MARROW', 'STEM CELL'
    ],

    # 介入性放射
    'INTERVENTIONAL': [
        '^IR ', 'IR EMBO', 'IR DRAIN', 'IR INSERT', 'IR ANGIO',
        'EMBOLIZATION', 'ANGIOGRAM', 'FISTULOGRAM', 'NEPHROSTOMY',
        'FLUOROSCOP', 'CT GUID', 'ULTRASOUND GUID'
    ],

    # 器官移植
    'TRANSPLANT': [
        'TRANSPLANT', 'DONOR', 'RECIPIENT', 'GRAFT.*ORGAN'
    ]
}

# 手術動作類型關鍵字
PROCEDURE_KEYWORDS = {
    # 開放性手術
    'OPEN': ['OPEN', 'LAPAROTOMY', 'THORACOTOMY', 'CRANIOTOMY'],

    # 微創手術
    'MINIMALLY_INVASIVE': [
        'LAPAROSCOP', 'ARTHROSCOP', 'ENDOSCOP', 'THORACOSCOP',
        'ROBOT', 'DA VINCI', 'VATS', 'PERCUTANEOUS'
    ],

    # 內視鏡
    'ENDOSCOPIC': [
        'EGD', 'COLONOSCOPY', 'BRONCHOSCOPY', 'CYSTOSCOPY',
        'HYSTEROSCOPY', 'URETEROSCOPY', 'ESOPHAGOSCOPY'
    ]
}

def classify_surgery(surgery_name):
    """
    根據手術名稱分類到標準 CPT/ICHI 類別

    Parameters:
    -----------
    surgery_name : str
        原始手術名稱

    Returns:
    --------
    dict : 包含分類結果
        - category: 主要系統分類
        - category_name: 分類中文名稱
        - procedure_type: 手術類型 (Open/Minimally Invasive/Endoscopic)
        - standardized_name: 標準化手術名稱
    """
    if pd.isna(surgery_name):
        return {
            'category': 'OTHER',
            'category_name': 'Other/Unclassified',
            'procedure_type': 'Unknown',
            'standardized_name': None
        }

    name = str(surgery_name).upper()

    # 1. 先判斷手術類型
    procedure_type = 'Open'  # 預設為開放性手術
    for ptype, keywords in PROCEDURE_KEYWORDS.items():
        for kw in keywords:
            if re.search(kw, name):
                if ptype == 'MINIMALLY_INVASIVE':
                    procedure_type = 'Minimally Invasive'
                elif ptype == 'ENDOSCOPIC':
                    procedure_type = 'Endoscopic'
                break

    # 2. 根據解剖部位分類
    matched_category = 'OTHER'
    max_score = 0

    for category, keywords in ANATOMY_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if re.search(kw, name):
                score += 1
        if score > max_score:
            max_score = score
            matched_category = category

    # 3. 生成標準化名稱
    standardized_name = standardize_surgery_name(surgery_name, matched_category)

    return {
        'category': matched_category,
        'category_name': SURGERY_CATEGORIES.get(matched_category, 'Other/Unclassified'),
        'procedure_type': procedure_type,
        'standardized_name': standardized_name
    }

def standardize_surgery_name(surgery_name, category):
    """
    將手術名稱標準化
    """
    if pd.isna(surgery_name):
        return None

    name = str(surgery_name).upper()

    # ============================================================
    # 標準化規則 (基於 CPT/ICHI 命名規範)
    # ============================================================

    standardization_rules = {
        # === 肌肉骨骼系統 ===
        r'ORIF.*FRACTURE.*FEMUR': 'ORIF - Femur Fracture',
        r'ORIF.*FRACTURE.*TIBIA': 'ORIF - Tibia Fracture',
        r'ORIF.*FRACTURE.*HUMERUS': 'ORIF - Humerus Fracture',
        r'ORIF.*FRACTURE.*RADIUS': 'ORIF - Radius Fracture',
        r'ORIF.*FRACTURE.*ULNA': 'ORIF - Ulna Fracture',
        r'ORIF.*FRACTURE.*ANKLE': 'ORIF - Ankle Fracture',
        r'ORIF.*ANKLE': 'ORIF - Ankle',
        r'ORIF.*FRACTURE.*ACETABULUM': 'ORIF - Acetabulum Fracture',
        r'ORIF.*FRACTURE.*PELVIS': 'ORIF - Pelvis Fracture',
        r'ORIF.*PELVIS': 'ORIF - Pelvis',
        r'ORIF.*FRACTURE.*MANDIBLE': 'ORIF - Mandible Fracture',
        r'ORIF.*FRACTURE.*FACIAL': 'ORIF - Facial Bone Fracture',
        r'ORIF.*FRACTURE.*CLAVICLE': 'ORIF - Clavicle Fracture',
        r'ORIF.*FRACTURE': 'ORIF - Fracture (Other)',
        r'ORIF.*WRIST': 'ORIF - Wrist',
        r'ORIF.*HIP': 'ORIF - Hip',
        r'^ORIF$': 'ORIF - Unspecified',

        r'ARTHROPLASTY.*KNEE': 'Arthroplasty - Total Knee',
        r'ARTHROPLASTY.*HIP': 'Arthroplasty - Total Hip',
        r'ARTHROPLASTY.*SHOULDER': 'Arthroplasty - Shoulder',
        r'ARTHROPLASTY.*ANKLE': 'Arthroplasty - Ankle',
        r'ARTHROPLASTY.*ELBOW': 'Arthroplasty - Elbow',
        r'ARTHROPLASTY.*FINGER': 'Arthroplasty - Finger',

        r'ARTHROSCOPY.*KNEE.*ACL': 'Arthroscopy - Knee ACL Reconstruction',
        r'ARTHROSCOPY.*KNEE.*MENISCECTOMY': 'Arthroscopy - Knee Meniscectomy',
        r'ARTHROSCOPY.*KNEE': 'Arthroscopy - Knee',
        r'ARTHROSCOPY.*SHOULDER.*ROTATOR': 'Arthroscopy - Shoulder Rotator Cuff Repair',
        r'ARTHROSCOPY.*SHOULDER': 'Arthroscopy - Shoulder',
        r'ARTHROSCOPY.*HIP': 'Arthroscopy - Hip',

        r'LAMINECTOMY.*LUMBAR': 'Laminectomy - Lumbar',
        r'LAMINECTOMY.*CERVICAL': 'Laminectomy - Cervical',
        r'LAMINECTOMY.*THORACIC': 'Laminectomy - Thoracic',
        r'LAMINECTOMY': 'Laminectomy - Spine',

        r'DISCECTOMY.*LUMBAR': 'Discectomy - Lumbar',
        r'DISCECTOMY.*CERVICAL': 'Discectomy - Cervical',
        r'DISCECTOMY': 'Discectomy - Spine',

        r'FUSION.*SPINE.*LUMBAR|FUSION.*LUMBAR': 'Spinal Fusion - Lumbar',
        r'FUSION.*SPINE.*CERVICAL|FUSION.*CERVICAL': 'Spinal Fusion - Cervical',
        r'FUSION.*SPINE.*THORACIC': 'Spinal Fusion - Thoracic',
        r'FUSION.*SPINE|SPINAL FUSION': 'Spinal Fusion',
        r'FUSION.*ANKLE': 'Arthrodesis - Ankle',
        r'FUSION.*FINGER': 'Arthrodesis - Finger',

        r'AMPUTATION.*ABOVE KNEE|AMPUTATION.*AKA': 'Amputation - Above Knee',
        r'AMPUTATION.*BELOW KNEE|AMPUTATION.*BKA': 'Amputation - Below Knee',
        r'AMPUTATION.*TOE': 'Amputation - Toe',
        r'AMPUTATION.*FOOT': 'Amputation - Foot',
        r'AMPUTATION.*FINGER': 'Amputation - Finger',
        r'AMPUTATION.*HAND': 'Amputation - Hand',
        r'AMPUTATION.*ELBOW': 'Amputation - Upper Extremity',
        r'AMPUTATION': 'Amputation - Other',

        r'IRRIGATION AND DEBRIDEMENT.*SPINE': 'Irrigation & Debridement - Spine',
        r'IRRIGATION AND DEBRIDEMENT.*LOWER': 'Irrigation & Debridement - Lower Extremity',
        r'IRRIGATION AND DEBRIDEMENT.*UPPER': 'Irrigation & Debridement - Upper Extremity',
        r'IRRIGATION AND DEBRIDEMENT.*HIP': 'Irrigation & Debridement - Hip',
        r'IRRIGATION AND DEBRIDEMENT.*KNEE': 'Irrigation & Debridement - Knee',
        r'IRRIGATION AND DEBRIDEMENT': 'Irrigation & Debridement',

        r'DEBRIDEMENT.*WOUND': 'Wound Debridement',
        r'DEBRIDEMENT.*SKIN.*GRAFT': 'Debridement with Skin Graft',
        r'DEBRIDEMENT': 'Debridement',

        r'REPAIR.*JOINT.*HIP': 'Repair - Hip Joint',
        r'REPAIR.*JOINT.*KNEE': 'Repair - Knee Joint',
        r'REPAIR.*TENDON': 'Tendon Repair',
        r'REPAIR.*FRACTURE': 'Fracture Repair',

        r'REMOVAL.*HARDWARE.*HIP': 'Hardware Removal - Hip',
        r'REMOVAL.*HARDWARE.*KNEE': 'Hardware Removal - Knee',
        r'REMOVAL.*HARDWARE.*LOWER': 'Hardware Removal - Lower Extremity',
        r'REMOVAL.*HARDWARE.*UPPER': 'Hardware Removal - Upper Extremity',
        r'REMOVAL.*HARDWARE': 'Hardware Removal',

        # === 消化系統 ===
        r'CHOLECYSTECTOMY.*LAPAROSCOP.*CHOLANGIOGRAM': 'Laparoscopic Cholecystectomy with Cholangiogram',
        r'CHOLECYSTECTOMY.*LAPAROSCOP': 'Laparoscopic Cholecystectomy',
        r'CHOLECYSTECTOMY.*ROBOT': 'Robotic Cholecystectomy',
        r'CHOLECYSTECTOMY': 'Cholecystectomy',

        r'APPENDECTOMY.*LAPAROSCOP': 'Laparoscopic Appendectomy',
        r'APPENDECTOMY': 'Appendectomy',

        r'COLECTOMY.*SIGMOID': 'Colectomy - Sigmoid',
        r'COLECTOMY.*RIGHT|COLECTOMY.*HEMICOLECTOMY.*RIGHT': 'Colectomy - Right Hemicolectomy',
        r'COLECTOMY.*LEFT|COLECTOMY.*HEMICOLECTOMY.*LEFT': 'Colectomy - Left Hemicolectomy',
        r'COLECTOMY.*TOTAL': 'Colectomy - Total',
        r'COLECTOMY.*LAPAROSCOP': 'Laparoscopic Colectomy',
        r'COLECTOMY': 'Colectomy',

        r'GASTRECTOMY.*SLEEVE.*LAPAROSCOP': 'Laparoscopic Sleeve Gastrectomy',
        r'GASTRECTOMY.*SLEEVE': 'Sleeve Gastrectomy',
        r'GASTRECTOMY.*SUBTOTAL': 'Subtotal Gastrectomy',
        r'GASTRECTOMY.*TOTAL': 'Total Gastrectomy',
        r'GASTRECTOMY': 'Gastrectomy',

        r'HERNIORRHAPHY.*INGUINAL': 'Hernia Repair - Inguinal',
        r'HERNIORRHAPHY.*VENTRAL': 'Hernia Repair - Ventral',
        r'HERNIORRHAPHY.*UMBILICAL': 'Hernia Repair - Umbilical',
        r'HERNIORRHAPHY.*INCISIONAL': 'Hernia Repair - Incisional',
        r'REPAIR.*HERNIA.*INGUINAL': 'Hernia Repair - Inguinal',
        r'REPAIR.*HERNIA.*VENTRAL': 'Hernia Repair - Ventral',
        r'REPAIR.*HERNIA.*HIATAL': 'Hernia Repair - Hiatal',
        r'REPAIR.*HERNIA.*UMBILICAL': 'Hernia Repair - Umbilical',
        r'REPAIR.*HERNIA': 'Hernia Repair',

        r'LAPAROTOMY.*EXPLORATORY': 'Exploratory Laparotomy',
        r'LAPAROTOMY': 'Laparotomy',

        r'LAPAROSCOPY.*DIAGNOSTIC': 'Diagnostic Laparoscopy',
        r'LAPAROSCOPY.*STAGING': 'Staging Laparoscopy',
        r'LAPAROSCOPY': 'Laparoscopy',

        r'EGD.*EUS.*FNA': 'EGD with EUS and FNA',
        r'EGD.*ESOPHAGOGASTRODUODENOSCOPY': 'Esophagogastroduodenoscopy (EGD)',
        r'EGD': 'Esophagogastroduodenoscopy (EGD)',

        r'COLONOSCOPY.*POLYPECTOMY': 'Colonoscopy with Polypectomy',
        r'COLONOSCOPY': 'Colonoscopy',

        r'RESECTION.*ABDOMINOPERINEAL': 'Abdominoperineal Resection',
        r'RESECTION.*RECTUM.*LOW ANTERIOR': 'Low Anterior Resection',
        r'RESECTION.*RECTUM': 'Rectal Resection',
        r'RESECTION.*SMALL BOWEL': 'Small Bowel Resection',
        r'RESECTION.*LIVER': 'Hepatic Resection',

        r'LOBECTOMY.*LIVER': 'Hepatic Lobectomy',

        r'WHIPPLE|PANCREATICODUODENECTOMY': 'Whipple Procedure',

        r'GASTROSTOMY.*PEG': 'PEG Tube Placement',
        r'GASTROSTOMY': 'Gastrostomy',

        r'ILEOSTOMY': 'Ileostomy Creation',
        r'COLOSTOMY': 'Colostomy Creation',
        r'CLOSURE.*ILEOSTOMY': 'Ileostomy Closure',
        r'CLOSURE.*COLOSTOMY': 'Colostomy Closure',

        # === 女性生殖系統 ===
        r'HYSTERECTOMY.*TOTAL.*ABDOMINAL.*ROBOT.*BSO': 'Robotic Total Abdominal Hysterectomy with BSO',
        r'HYSTERECTOMY.*TOTAL.*ABDOMINAL.*LAPAROSCOP.*BSO': 'Laparoscopic Total Hysterectomy with BSO',
        r'HYSTERECTOMY.*TOTAL.*ABDOMINAL.*BSO.*DEBULKING': 'Total Hysterectomy with BSO and Debulking',
        r'HYSTERECTOMY.*TOTAL.*ABDOMINAL.*LYMPHADENECTOMY': 'Total Hysterectomy with Lymphadenectomy',
        r'HYSTERECTOMY.*TOTAL.*ABDOMINAL.*BSO': 'Total Abdominal Hysterectomy with BSO',
        r'HYSTERECTOMY.*TOTAL.*ABDOMINAL': 'Total Abdominal Hysterectomy',
        r'HYSTERECTOMY.*TOTAL.*VAGINAL': 'Total Vaginal Hysterectomy',
        r'HYSTERECTOMY.*VAGINAL.*ROBOT': 'Robotic Vaginal Hysterectomy',
        r'HYSTERECTOMY.*VAGINAL.*LAPAROSCOP': 'Laparoscopic-Assisted Vaginal Hysterectomy',
        r'HYSTERECTOMY.*VAGINAL': 'Vaginal Hysterectomy',
        r'HYSTERECTOMY.*ABDOMINAL.*LAPAROSCOP': 'Laparoscopic Hysterectomy',
        r'HYSTERECTOMY': 'Hysterectomy',

        r'DILATION AND EVACUATION.*UTERUS': 'Dilation and Evacuation (D&E)',
        r'DILATION AND CURETTAGE': 'Dilation and Curettage (D&C)',

        r'HYSTEROSCOPY.*BIOPSY|HYSTEROSCOPY.*POLYPECTOMY': 'Hysteroscopy with Biopsy/Polypectomy',
        r'HYSTEROSCOPY.*MYOMECTOMY': 'Hysteroscopy with Myomectomy',
        r'HYSTEROSCOPY.*D.*C': 'Hysteroscopy with D&C',
        r'HYSTEROSCOPY': 'Hysteroscopy',

        r'OOPHORECTOMY': 'Oophorectomy',
        r'SALPINGO-OOPHORECTOMY': 'Salpingo-Oophorectomy',
        r'MYOMECTOMY': 'Myomectomy',
        r'CESAREAN.*SECTION|C-SECTION': 'Cesarean Section',

        # === 泌尿系統 ===
        r'CYSTOSCOPY.*RETROGRADE.*URETEROSCOPY.*LITHOTRIPSY.*STENT': 'Cystoscopy with Ureteroscopy, Lithotripsy and Stent',
        r'CYSTOSCOPY.*URETEROSCOPY.*LITHOTRIPSY': 'Cystoscopy with Ureteroscopy and Lithotripsy',
        r'CYSTOSCOPY.*RETROGRADE.*PYELOGRAM': 'Cystoscopy with Retrograde Pyelogram',
        r'CYSTOSCOPY.*BIOPSY': 'Cystoscopy with Biopsy',
        r'CYSTOSCOPY.*STENT': 'Cystoscopy with Stent Placement',
        r'CYSTOSCOPY': 'Cystoscopy',

        r'NEPHRECTOMY.*PARTIAL.*LAPAROSCOP': 'Laparoscopic Partial Nephrectomy',
        r'NEPHRECTOMY.*PARTIAL.*ROBOT': 'Robotic Partial Nephrectomy',
        r'NEPHRECTOMY.*PARTIAL': 'Partial Nephrectomy',
        r'NEPHRECTOMY.*RADICAL.*LAPAROSCOP': 'Laparoscopic Radical Nephrectomy',
        r'NEPHRECTOMY.*RADICAL': 'Radical Nephrectomy',
        r'NEPHRECTOMY.*DONOR': 'Donor Nephrectomy',
        r'NEPHRECTOMY.*LAPAROSCOP': 'Laparoscopic Nephrectomy',
        r'NEPHRECTOMY': 'Nephrectomy',

        r'TURBT': 'Transurethral Resection of Bladder Tumor (TURBT)',
        r'TRANSURETHRAL RESECTION.*PROSTATE|TURP': 'Transurethral Resection of Prostate (TURP)',

        r'PERCUTANEOUS NEPHROLITHOTOMY': 'Percutaneous Nephrolithotomy (PCNL)',
        r'URETEROSCOPY.*LITHOTRIPSY': 'Ureteroscopy with Lithotripsy',
        r'URETEROSCOPY': 'Ureteroscopy',

        # === 男性生殖系統 ===
        r'DA VINCI.*PROSTATECTOMY|PROSTATECTOMY.*ROBOT': 'Robotic Radical Prostatectomy',
        r'PROSTATECTOMY.*RADICAL.*LAPAROSCOP': 'Laparoscopic Radical Prostatectomy',
        r'PROSTATECTOMY.*RADICAL': 'Radical Prostatectomy',
        r'PROSTATECTOMY': 'Prostatectomy',

        r'ORCHIECTOMY': 'Orchiectomy',
        r'HYDROCELECTOMY': 'Hydrocelectomy',

        # === 神經系統 ===
        r'CRANIOTOMY.*TUMOR': 'Craniotomy for Tumor Excision',
        r'CRANIOTOMY.*ANEURYSM': 'Craniotomy for Aneurysm',
        r'CRANIOTOMY.*HEMATOMA': 'Craniotomy for Hematoma Evacuation',
        r'CRANIOTOMY': 'Craniotomy',

        r'CRANIECTOMY.*DECOMPRESSIVE': 'Decompressive Craniectomy',
        r'CRANIECTOMY': 'Craniectomy',

        r'CRANIOPLASTY': 'Cranioplasty',

        r'VP SHUNT|VENTRICULOPERITONEAL SHUNT': 'VP Shunt Placement',
        r'SHUNT.*REVISION': 'Shunt Revision',

        r'INSERTION.*DEEP BRAIN STIMULATOR': 'Deep Brain Stimulator Insertion',
        r'REPLACEMENT.*DEEP BRAIN STIMULATOR': 'Deep Brain Stimulator Replacement',

        r'DECOMPRESSION.*CARPAL TUNNEL': 'Carpal Tunnel Release',
        r'DECOMPRESSION.*ULNAR NERVE': 'Ulnar Nerve Decompression',

        # === 心血管系統 ===
        r'CABG.*CORONARY ARTERY BYPASS': 'Coronary Artery Bypass Graft (CABG)',
        r'CABG': 'Coronary Artery Bypass Graft (CABG)',

        r'REPLACEMENT.*VALVE.*AORTIC': 'Aortic Valve Replacement',
        r'REPLACEMENT.*VALVE.*MITRAL': 'Mitral Valve Replacement',
        r'REPAIR.*VALVE.*MITRAL': 'Mitral Valve Repair',
        r'TAVR|TRANSCATHETER.*AORTIC.*VALVE': 'Transcatheter Aortic Valve Replacement (TAVR)',

        r'REPAIR.*AORTA.*THORACIC.*ENDOVASCULAR': 'Thoracic Endovascular Aortic Repair (TEVAR)',
        r'REPAIR.*AORTA.*ABDOMINAL.*ENDOVASCULAR|EVAR': 'Endovascular Aneurysm Repair (EVAR)',
        r'REPAIR.*ANEURYSM.*AORTIC': 'Aortic Aneurysm Repair',

        r'CREATION.*AV FISTULA': 'AV Fistula Creation',
        r'AV FISTULOGRAM': 'AV Fistulogram',

        r'ENDARTERECTOMY.*CAROTID': 'Carotid Endarterectomy',
        r'ENDARTERECTOMY': 'Endarterectomy',

        r'INSERTION.*PACEMAKER': 'Pacemaker Insertion',
        r'INSERTION.*ICD|INSERTION.*DEFIBRILLATOR': 'ICD Insertion',
        r'REPLACEMENT.*PACEMAKER': 'Pacemaker Replacement',

        r'ANGIOPLASTY.*CORONARY': 'Coronary Angioplasty',
        r'ANGIOPLASTY': 'Angioplasty',

        # === 呼吸系統 ===
        r'LOBECTOMY.*LUNG.*VATS': 'VATS Lobectomy',
        r'LOBECTOMY.*LUNG': 'Pulmonary Lobectomy',
        r'LOBECTOMY': 'Lobectomy',

        r'WEDGE RESECTION.*LUNG': 'Lung Wedge Resection',

        r'PNEUMONECTOMY': 'Pneumonectomy',

        r'BRONCHOSCOPY.*BIOPSY': 'Bronchoscopy with Biopsy',
        r'BRONCHOSCOPY.*LAVAGE': 'Bronchoscopy with BAL',
        r'BRONCHOSCOPY': 'Bronchoscopy',

        r'TRACHEOSTOMY': 'Tracheostomy',
        r'THORACOTOMY': 'Thoracotomy',

        r'VIDEO-ASSISTED THORACOSCOPIC|VATS': 'Video-Assisted Thoracoscopic Surgery (VATS)',

        # === 內分泌系統 ===
        r'THYROIDECTOMY.*TOTAL': 'Total Thyroidectomy',
        r'THYROIDECTOMY.*SUBTOTAL|THYROIDECTOMY.*PARTIAL': 'Partial Thyroidectomy',
        r'THYROIDECTOMY': 'Thyroidectomy',

        r'PARATHYROIDECTOMY': 'Parathyroidectomy',
        r'ADRENALECTOMY.*LAPAROSCOP': 'Laparoscopic Adrenalectomy',
        r'ADRENALECTOMY': 'Adrenalectomy',

        # === 眼科 ===
        r'VITRECTOMY': 'Vitrectomy',
        r'CATARACT.*EXTRACTION|PHACOEMULSIFICATION': 'Cataract Surgery',
        r'BLEPHAROPLASTY': 'Blepharoplasty',
        r'STRABISMUS.*SURGERY': 'Strabismus Surgery',

        # === 耳科 ===
        r'COCHLEAR IMPLANT': 'Cochlear Implant',
        r'MASTOIDECTOMY': 'Mastoidectomy',
        r'TYMPANOPLASTY': 'Tympanoplasty',
        r'MYRINGOTOMY': 'Myringotomy',

        # === 移植 ===
        r'TRANSPLANT.*RECIPIENT.*KIDNEY.*LIVING': 'Kidney Transplant - Living Donor',
        r'TRANSPLANT.*RECIPIENT.*KIDNEY.*DECEASED': 'Kidney Transplant - Deceased Donor',
        r'TRANSPLANT.*RECIPIENT.*KIDNEY': 'Kidney Transplant',
        r'TRANSPLANT.*RECIPIENT.*LIVER': 'Liver Transplant',
        r'TRANSPLANT.*RECIPIENT.*HEART': 'Heart Transplant',
        r'TRANSPLANT.*RECIPIENT.*LUNG': 'Lung Transplant',
        r'TRANSPLANT.*RECIPIENT.*PANCREAS': 'Pancreas Transplant',
        r'TRANSPLANT.*DONOR.*KIDNEY': 'Donor Nephrectomy',
        r'TRANSPLANT.*DONOR.*LIVER': 'Donor Hepatectomy',

        # === 介入性放射 ===
        r'IR EMBO.*NON-CNS': 'IR Embolization - Non-CNS',
        r'IR EMBO.*CNS': 'IR Embolization - CNS',
        r'IR EMBO': 'IR Embolization',
        r'IR DRAIN': 'IR Drainage',
        r'IR INSERT': 'IR Catheter Insertion',
        r'IR ANGIO': 'IR Angiography',

        r'ANGIOGRAM.*CEREBRAL.*EMBOLIZATION': 'Cerebral Angiogram with Embolization',
        r'ANGIOGRAM.*CEREBRAL': 'Cerebral Angiogram',
        r'ANGIOGRAM.*SPINAL': 'Spinal Angiogram',
        r'ANGIOGRAM.*VISCERAL': 'Visceral Angiogram',
        r'ANGIOGRAM.*LOWER.*ANGIOPLASTY': 'Lower Extremity Angiogram with Angioplasty',
        r'ANGIOGRAM': 'Angiogram',

        # === 其他 ===
        r'EXAM UNDER ANESTHESIA.*ANORECTAL': 'EUA - Anorectal',
        r'EXAM UNDER ANESTHESIA': 'Exam Under Anesthesia',

        r'TONSILLECTOMY.*ADENOIDECTOMY': 'Tonsillectomy and Adenoidectomy',
        r'TONSILLECTOMY': 'Tonsillectomy',
        r'ADENOIDECTOMY': 'Adenoidectomy',

        r'BIOPSY.*BRAIN': 'Brain Biopsy',
        r'BIOPSY.*LIVER': 'Liver Biopsy',
        r'BIOPSY.*LUNG': 'Lung Biopsy',
        r'BIOPSY.*PROSTATE': 'Prostate Biopsy',
        r'BIOPSY.*BREAST': 'Breast Biopsy',
        r'BIOPSY.*LYMPH NODE': 'Lymph Node Biopsy',
        r'BIOPSY.*BONE': 'Bone Biopsy',
        r'BIOPSY.*MUSCLE': 'Muscle Biopsy',
        r'BIOPSY': 'Biopsy',

        r'EXCISION.*LESION': 'Lesion Excision',
        r'EXCISION.*TUMOR': 'Tumor Excision',
        r'EXCISION.*CYST': 'Cyst Excision',
        r'EXCISION': 'Excision',

        r'EXPLORATION.*NECK': 'Neck Exploration',
        r'EXPLORATION.*ABDOMEN': 'Abdominal Exploration',
        r'EXPLORATION': 'Exploration',

        r'MASTECTOMY.*BILATERAL': 'Bilateral Mastectomy',
        r'MASTECTOMY.*MODIFIED RADICAL': 'Modified Radical Mastectomy',
        r'MASTECTOMY.*SIMPLE': 'Simple Mastectomy',
        r'MASTECTOMY': 'Mastectomy',

        r'BREAST RECONSTRUCTION': 'Breast Reconstruction',
        r'LUMPECTOMY': 'Lumpectomy',
    }

    # 嘗試匹配規則
    for pattern, standardized in standardization_rules.items():
        if re.search(pattern, name):
            return standardized

    # 如果沒有匹配，返回簡化的名稱
    # 取第一個逗號前的內容作為主要手術
    main_part = name.split(',')[0].strip()
    return main_part


def process_surgery_column(df, surgery_col='Surgery_Name'):
    """
    處理整個 DataFrame 的手術名稱欄位

    Parameters:
    -----------
    df : DataFrame
        包含手術名稱的 DataFrame
    surgery_col : str
        手術名稱欄位名

    Returns:
    --------
    DataFrame : 新增分類欄位的 DataFrame
    """
    # 應用分類
    results = df[surgery_col].apply(classify_surgery)

    # 展開結果到新欄位
    df['Surgery_Category'] = results.apply(lambda x: x['category'])
    df['Surgery_Category_Name'] = results.apply(lambda x: x['category_name'])
    df['Surgery_Procedure_Type'] = results.apply(lambda x: x['procedure_type'])
    df['Surgery_Standardized'] = results.apply(lambda x: x['standardized_name'])

    return df


if __name__ == '__main__':
    # 測試
    test_surgeries = [
        'HYSTERECTOMY, TOTAL, ABDOMINAL, ROBOT-ASSISTED, LAPAROSCOPIC, WITH BSO',
        'CHOLECYSTECTOMY, LAPAROSCOPIC',
        'ORIF, FRACTURE, FEMUR',
        'CYSTOSCOPY, WITH RETROGRADE PYELOGRAM, URETEROSCOPY, URINARY CALCULUS LASER LITHOTRIPSY, + STENT INSERT',
        'CRANIOTOMY, FOR TUMOR',
        'CABG (CORONARY ARTERY BYPASS GRAFT)',
        'TRANSPLANT RECIPIENT, KIDNEY, FROM LIVING DONOR',
        'IR EMBO NON-CNS',
    ]

    print('=== 手術分類測試 ===\n')
    for surgery in test_surgeries:
        result = classify_surgery(surgery)
        print(f'原始: {surgery}')
        print(f'  類別: {result["category_name"]}')
        print(f'  類型: {result["procedure_type"]}')
        print(f'  標準化: {result["standardized_name"]}')
        print()
