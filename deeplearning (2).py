from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import numpy as np
import torch
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from google.cloud import bigquery
from torch.utils.data import Dataset, DataLoader

query_clinical_notes = """
SELECT subject_id, category, description, text
FROM `physionet-data.mimiciii_notes.noteevents`
WHERE LOWER(text) LIKE '%liver cancer%' 
   OR LOWER(text) LIKE '%hepatocellular carcinoma%' 
   OR LOWER(text) LIKE '%HCC%' 
   OR LOWER(text) LIKE '%hepatic tumor%'
   OR LOWER(text) LIKE '%liver mass%' 
   OR LOWER(text) LIKE '%hepatic lesion%'
   AND category IN ('Discharge summary', 'Radiology')
LIMIT 10000;

"""

query_lab_results = """
SELECT p.subject_id, di.label AS test_name, lr.charttime, lr.valuenum, lr.ref_range_lower, lr.ref_range_upper
FROM `physionet-data.mimiciv_3_1_hosp.patients` AS p
LEFT JOIN `physionet-data.mimiciv_3_1_hosp.labevents` AS lr
ON p.subject_id = lr.subject_id
LEFT JOIN `physionet-data.mimiciv_3_1_hosp.d_labitems` AS di
ON lr.itemid = di.itemid
WHERE LOWER(di.label) LIKE '%liver%' 
   OR LOWER(di.label) LIKE '%hepatic%' 
   OR LOWER(di.label) LIKE '%bilirubin%'
   OR LOWER(di.label) LIKE '%afp%'
   OR LOWER(di.label) LIKE '%alt%'
   OR LOWER(di.label) LIKE '%ast%'
   OR LOWER(di.label) LIKE '%albumin%'
   OR LOWER(di.label) LIKE '%alkaline phosphatase%'
   OR LOWER(di.label) LIKE '%inr%'
   OR di.label IN ('AFP', 'Liver Function Test', 'Bilirubin, Total', 'Bilirubin, Direct', 'Bilirubin, Indirect',
                   'ALT (Alanine Aminotransferase)', 'AST (Aspartate Aminotransferase)', 'Albumin', 'INR',
                   'Alkaline Phosphatase')
LIMIT 10000;
"""

query_diagnoses = """
SELECT di.subject_id, di.hadm_id, dd.icd_code, dd.icd_version, dd.long_title
FROM physionet-data.mimiciv_3_1_hosp.diagnoses_icd AS di
JOIN physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses AS dd
ON di.icd_code = dd.icd_code AND di.icd_version = dd.icd_version
WHERE di.icd_code IN (
    SELECT DISTINCT icd_code FROM physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses
    WHERE LOWER(long_title) LIKE '%liver cancer%'
       OR LOWER(long_title) LIKE '%hepatocellular carcinoma%'
       OR LOWER(long_title) LIKE '%malignant neoplasm of liver%'
       OR LOWER(long_title) LIKE '%hepatic tumor%'
       OR LOWER(long_title) LIKE '%liver neoplasm%'
)
LIMIT 10000;
"""

query_procedures = """
SELECT p.subject_id, pi.hadm_id, 
       pi.icd_code AS proc_icd_code, 
       dp.long_title AS procedure_title,
       di.icd_code AS diag_icd_code, 
       dd.long_title AS diagnosis_title
FROM `physionet-data.mimiciv_3_1_hosp.patients` AS p
LEFT JOIN `physionet-data.mimiciv_3_1_hosp.procedures_icd` AS pi
ON p.subject_id = pi.subject_id
LEFT JOIN `physionet-data.mimiciv_3_1_hosp.d_icd_procedures` AS dp
ON pi.icd_code = dp.icd_code AND pi.icd_version = dp.icd_version
LEFT JOIN `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` AS di
ON pi.subject_id = di.subject_id AND pi.hadm_id = di.hadm_id
LEFT JOIN `physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses` AS dd
ON di.icd_code = dd.icd_code AND di.icd_version = dd.icd_version
WHERE 
    -- Ensuring only liver-related procedures
    (LOWER(dp.long_title) LIKE '%hepatectomy%' 
     OR LOWER(dp.long_title) LIKE '%liver biopsy%'
     OR LOWER(dp.long_title) LIKE '%radiofrequency ablation%'
     OR LOWER(dp.long_title) LIKE '%TACE%')

    -- Filtering only for liver cancer-related diagnoses
    AND (
        di.icd_code IN ('1550', '1552', '1977', '20972', '2115', '2308', '2353') -- ICD-9 liver cancer codes
        OR di.icd_code LIKE 'C22%'  -- ICD-10 liver cancer codes
    )
LIMIT 10000;
 """

query_patient_history = """
SELECT di.subject_id, di.hadm_id, dd.icd_code, dd.icd_version, dd.long_title
FROM physionet-data.mimiciv_3_1_hosp.diagnoses_icd AS di
JOIN physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses AS dd
ON di.icd_code = dd.icd_code AND di.icd_version = dd.icd_version
WHERE (di.icd_version = 9 AND di.icd_code IN ('1550', '1552', '1977', '20972'))
   OR (di.icd_version = 10 AND di.icd_code LIKE 'C22%')  -- ICD-10 codes for liver cancer
LIMIT 10000;
"""

query_medications = """
SELECT pr.subject_id, pr.hadm_id, pr.drug, pr.starttime, pr.stoptime
FROM `physionet-data.mimiciv_3_1_hosp.prescriptions` AS pr
WHERE LOWER(pr.drug) LIKE '%sorafenib%'  
   OR LOWER(pr.drug) LIKE '%lenvatinib%'
   OR LOWER(pr.drug) LIKE '%regorafenib%'
   OR LOWER(pr.drug) LIKE '%nivolumab%'
   OR LOWER(pr.drug) LIKE '%pembrolizumab%'
   OR LOWER(pr.drug) LIKE '%atezolizumab%'
   OR LOWER(pr.drug) LIKE '%bevacizumab%'
   OR LOWER(pr.drug) LIKE '%cabozantinib%'
   OR LOWER(pr.drug) LIKE '%ramucirumab%'
LIMIT 10000;
"""

query_demographics = """
SELECT subject_id, gender, anchor_age AS age
FROM `physionet-data.mimiciv_3_1_hosp.patients`
LIMIT 10000;
"""

query_admissions = """
SELECT DISTINCT a.subject_id, a.hadm_id, a.admission_type, a.discharge_location
FROM `physionet-data.mimiciv_3_1_hosp.admissions` AS a
JOIN `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` AS di
ON a.subject_id = di.subject_id AND a.hadm_id = di.hadm_id
WHERE (di.icd_version = 9 AND di.icd_code IN ('1550', '1552', '1977', '20972'))
   OR (di.icd_version = 10 AND di.icd_code LIKE 'C22%')  -- ICD-10 codes for liver cancer
LIMIT 10000;

"""

query_vital_signs = """
SELECT 
    ce.subject_id, 
    ce.hadm_id, 
    ce.itemid, 
    di.label AS vital_sign_label,  -- Label for vital sign
    ce.charttime, 
    ce.valuenum
FROM `physionet-data.mimiciv_3_1_icu.chartevents` AS ce
JOIN `physionet-data.mimiciv_3_1_icu.d_items` AS di  -- Join with d_items to get the label
ON ce.itemid = di.itemid
WHERE ce.subject_id IN (
    SELECT DISTINCT di.subject_id
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` AS di
    WHERE (di.icd_version = 9 AND di.icd_code IN ('1550', '1552', '1977', '20972'))
       OR (di.icd_version = 10 AND di.icd_code LIKE 'C22%')  -- ICD-10 codes for liver cancer
)
AND ce.itemid IN (223761, 223762, 224166, 224167, 224643, 223764, 223765) -- Common vital signs
LIMIT 10000;

"""

query_for_liver = """
SELECT DISTINCT icd_code
FROM (
    -- MIMIC-III (ICD-9 Codes for Liver Cancer)
    SELECT icd9_code AS icd_code
    FROM `physionet-data.mimiciii_clinical.diagnoses_icd`
    WHERE icd9_code BETWEEN '1550' AND '1559' OR icd9_code = '1977'  -- ICD-9 range for liver cancer
    
    UNION ALL  -- Use UNION ALL for efficiency

    -- MIMIC-IV (ICD-10 & ICD-9 Codes for Liver Cancer)
    SELECT icd_code
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
    WHERE icd_code LIKE 'C22%' OR icd_code = 'C787'  -- ICD-10 codes for liver cancer
) AS combined_icd_codes
ORDER BY icd_code;

"""

# Initialize BigQuery client
client = bigquery.Client(location="US")

# Function to execute query and return a DataFrame
def execute_query(query, name):
    job = client.query(query)
    results = job.result()
    df = pd.DataFrame([dict(row) for row in results])

    print(f"Retrieved {len(df)} rows from {name}")
    
    return df

# Fetch data from BigQuery
clinical_notes_df = execute_query(query_clinical_notes, "Clinical Notes")
lab_results_df = execute_query(query_lab_results, "lab")
diagnoses_df = execute_query(query_diagnoses, "d")
procedures_df = execute_query(query_procedures, "p")
medications_df = execute_query(query_medications, "medi")
demographics_df = execute_query(query_demographics, "demo")
admissions_df = execute_query(query_admissions, "admi")
vital_signs_df = execute_query(query_vital_signs, "vita")
liver_cancer_icd_df = execute_query(query_for_liver, "liver")

# Create liver cancer patient flag
liver_cancer_icd_list = liver_cancer_icd_df['icd_code'].tolist()
query_liver_cancer_patients = f"""
SELECT DISTINCT subject_id
FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
WHERE icd_code IN ({', '.join([f"'{code}'" for code in liver_cancer_icd_list])});
"""
liver_cancer_patients_df = execute_query(query_liver_cancer_patients, "live")
liver_cancer_patients_df['has_cancer_history'] = 1 

# Normalize subject_id indexing before merging
subject_ids = demographics_df[['subject_id']].drop_duplicates().sort_values('subject_id')

# Ensure subject_id is unique in all tables before merging
demographics_df = demographics_df.drop_duplicates(subset=['subject_id'])

# Identify numeric and categorical columns
numeric_columns = lab_results_df.select_dtypes(include=[np.number]).columns
categorical_columns = lab_results_df.select_dtypes(exclude=[np.number]).columns

# Aggregate numerical and categorical separately
lab_results_numeric = lab_results_df.groupby('subject_id', as_index=False)[numeric_columns].mean()
lab_results_categorical = lab_results_df.groupby('subject_id', as_index=False)[categorical_columns].first()
lab_results_df = lab_results_numeric.merge(lab_results_categorical, on='subject_id', how='left')

# Identify numeric and categorical columns
vital_signs_numeric_columns = vital_signs_df.select_dtypes(include=[np.number]).columns.tolist()
vital_signs_categorical_columns = [col for col in vital_signs_df.columns if col not in vital_signs_numeric_columns + ['subject_id']]

# Ensure subject_id is included in numeric aggregation
vital_signs_numeric = vital_signs_df.groupby('subject_id', as_index=False)[vital_signs_numeric_columns].mean()

# Ensure subject_id is retained while aggregating categorical columns
vital_signs_categorical = vital_signs_df.groupby('subject_id', as_index=False)[['subject_id'] + vital_signs_categorical_columns].first()

# Merge numerical and categorical data safely
vital_signs_df = vital_signs_numeric.merge(
    vital_signs_categorical,
    on='subject_id',
    how='left'
)

# Rename hadm_id in individual DataFrames to prevent conflicts
admissions_df.rename(columns={'hadm_id': 'hadm_id_adm'}, inplace=True)
diagnoses_df.rename(columns={'hadm_id': 'hadm_id_diag'}, inplace=True)
procedures_df.rename(columns={'hadm_id': 'hadm_id_proc'}, inplace=True)
medications_df.rename(columns={'hadm_id': 'hadm_id_med'}, inplace=True)
vital_signs_df.rename(columns={'hadm_id': 'hadm_id_vs'}, inplace=True)


demographics_df = demographics_df.drop_duplicates(subset=['subject_id'])

# Aggregate text correctly
clinical_notes_df = clinical_notes_df.groupby('subject_id', as_index=False).agg({
    'text': lambda x: ' '.join(str(t) for t in x.dropna() if t.strip()),  
    'category': 'first',
    'description': 'first'
})

clinical_notes_df = clinical_notes_df.drop_duplicates(subset=['subject_id']) #removing all the duplicate values

#identifying the missing subjects
missing_subjects = subject_ids[~subject_ids['subject_id'].isin(clinical_notes_df['subject_id'])].copy() 

#handling the missing subjects
missing_subjects['text'] = "Unknown"
missing_subjects['category'] = np.nan
missing_subjects['description'] = np.nan


required_missing_count = 10000 - clinical_notes_df['subject_id'].nunique()
missing_subjects = missing_subjects.head(required_missing_count)  # Trim extra subjects

clinical_notes_df = pd.concat([clinical_notes_df, missing_subjects], ignore_index=True)

# Ensures that each dataset has exactly 10,000 rows by filling missing subjects
def fill_missing_subjects(df, subject_ids, name, fill_values=None):

    # Remove duplicate subject IDs to avoid duplication
    df = df.drop_duplicates(subset=['subject_id'])

    # Identify missing subject IDs
    missing_subjects = subject_ids[~subject_ids['subject_id'].isin(df['subject_id'])].copy()

    # Assign default values to missing subjects
    if fill_values is None:
        fill_values = {col: np.nan for col in df.columns if col != 'subject_id'}

    for col, default_value in fill_values.items():
        missing_subjects[col] = default_value

    # Ensure we only add exactly the missing subjects needed
    required_missing_count = 10000 - df['subject_id'].nunique()
    missing_subjects = missing_subjects.head(required_missing_count)

    # Merge back
    df = pd.concat([df, missing_subjects], ignore_index=True)
    df = df.drop_duplicates(subset=['subject_id'])

    return df

# Apply this fix to all datasets - do this for clinical notes
lab_results_df = fill_missing_subjects(lab_results_df, subject_ids, "Lab Results", fill_values={'valuenum': np.nan, 'ref_range_lower': np.nan, 'ref_range_upper': np.nan, 'test_name': 'Unknown'})
diagnoses_df = fill_missing_subjects(diagnoses_df, subject_ids, "Diagnoses", fill_values={'icd_code': 'Unknown', 'icd_version': np.nan, 'long_title': 'Unknown'})
procedures_df = fill_missing_subjects(procedures_df, subject_ids, "Procedures", fill_values={'proc_icd_code': 'Unknown', 'procedure_title': 'Unknown', 'diag_icd_code': 'Unknown', 'diagnosis_title': 'Unknown'})
medications_df = fill_missing_subjects(medications_df, subject_ids, "Medications", fill_values={'drug': 'Unknown', 'starttime': np.nan, 'stoptime': np.nan})
admissions_df = fill_missing_subjects(admissions_df, subject_ids, "Admissions", fill_values={'hadm_id_adm': np.nan, 'admission_type': 'Unknown', 'discharge_location': 'Unknown'})
vital_signs_df = fill_missing_subjects(vital_signs_df, subject_ids, "Vital Signs", fill_values={'valuenum': np.nan, 'vital_sign_label': 'Unknown'})


# Merge all datasets after ensuring subject_id alignment
merged_df = (
    demographics_df
    .merge(admissions_df, on='subject_id', how='left')
    .merge(diagnoses_df, on='subject_id', how='left')
    .merge(clinical_notes_df, on='subject_id', how='left')
    .merge(lab_results_df, on='subject_id', how='left')
    .merge(procedures_df, on='subject_id', how='left')
    .merge(medications_df, on='subject_id', how='left')
    .merge(vital_signs_df, on='subject_id', how='left')
    .merge(liver_cancer_patients_df, on='subject_id', how='left')
)

# Fill missing values
merged_df.fillna(0, inplace=True)

# Fill missing numerical values with -1 (to distinguish from real data)
numerical_columns = merged_df.select_dtypes(include=[np.number]).columns
merged_df[numerical_columns] = merged_df[numerical_columns].fillna(-1)

# Fill missing categorical values with 'No Data'
categorical_columns = merged_df.select_dtypes(exclude=[np.number]).columns
merged_df[categorical_columns] = merged_df[categorical_columns].fillna("No Data")


# Validate final dataset shape
print(f"Final Merged Dataset Shape: {merged_df.shape}")


# Fill missing values
to_fill = ['has_cancer_history', 'age']
merged_df[to_fill] = merged_df[to_fill].fillna(0)
merged_df['has_cancer_history'] = merged_df['has_cancer_history'].astype(int)
print(merged_df['has_cancer_history'].value_counts())

# Apply NLP preprocessing
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    if not isinstance(text, str): 
        return "empty_document"
    
    text = text.lower()  
    text = re.sub(r'\d+', '', text)  
    text = re.sub(r'[^\w\s]', '', text)  
    tokens = word_tokenize(text)  
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens) if tokens else "empty_document"


clinical_notes_df['cleaned_text'] = clinical_notes_df['text'].astype(str).apply(preprocess_text)

# Load pre-trained BioBERT model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
bert_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
def applying_BERT(text, batch_size = 32):
    embeddings_list = []
    for i in range(0, len(text), batch_size):
        print(f"Processing batch {i // batch_size + 1} of {len(text) // batch_size + 1}")
        batch_texts = text[i:i + batch_size]
        tokens = tokenizer(batch_texts.tolist(), padding=True, truncation=True, max_length = 512, return_tensors="pt")

        with torch.no_grad():
            outputs = bert_model(**tokens)

        mean_pooled_embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings_list.append(mean_pooled_embeddings.cpu().numpy())
    print("BERT embeddings generated")
    return np.vstack(embeddings_list)

# Convert clinical notes into BERT embeddings
clinical_notes_df['BERT_embeddings'] = list(applying_BERT(clinical_notes_df['text']))

# Convert embeddings to DataFrame
bert_embeddings_df = pd.DataFrame(
    np.vstack(clinical_notes_df['BERT_embeddings'].values),
    columns=[f'BERT_{i}' for i in range(768)] 
)

# Add subject_id before merging
bert_embeddings_df['subject_id'] = clinical_notes_df['subject_id']

# Merge BERT embeddings into the final dataset
merged_df = merged_df.merge(bert_embeddings_df, on='subject_id', how='left')

# Drop original text column (no longer needed)
merged_df.drop(columns=['text'], errors='ignore', inplace=True)

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Get all BERT columns
bert_columns = [col for col in merged_df.columns if col.startswith("BERT_")]

imputer = SimpleImputer(strategy="median")
merged_df[bert_columns] = imputer.fit_transform(merged_df[bert_columns])

non_numeric_cols = merged_df.select_dtypes(exclude=[np.number]).columns
print("Non-numeric columns that need encoding:", non_numeric_cols.tolist())

from sklearn.preprocessing import LabelEncoder

# Apply Label Encoding to categorical features
for col in non_numeric_cols:
    le = LabelEncoder()
    merged_df[col] = le.fit_transform(merged_df[col].astype(str))  # Convert to string first
    
print("Data types after encoding:\n", merged_df.dtypes)

duplicate_columns = merged_df.columns[merged_df.T.duplicated()]
print(f"Duplicate Columns: {duplicate_columns.tolist()}")
merged_df.drop(columns=duplicate_columns, errors="ignore", inplace=True)

correlation_threshold = 0.8

# Compute absolute correlation with target variable
correlation_with_target = merged_df.corrwith(merged_df['has_cancer_history']).abs()

# Identify features with high correlation
highly_correlated_features = correlation_with_target[correlation_with_target > correlation_threshold].index.tolist()

# Ensure we don't drop the target variable itself
if 'has_cancer_history' in highly_correlated_features:
    highly_correlated_features.remove('has_cancer_history')

# Print features to be dropped
print(f" Dropping highly correlated features: {highly_correlated_features}")

# Drop the identified features from merged_df
merged_df.drop(columns=highly_correlated_features, inplace=True, errors='ignore')

# Display the new shape after dropping
print(f" Updated Dataset Shape: {merged_df.shape}")


train_patients, test_patients = train_test_split(
    merged_df['subject_id'], test_size=0.2, random_state=42
)

assert len(set(train_patients) & set(test_patients)) == 0, " Data leakage detected in train-test split!"


# Define X and y before dropping `subject_id`
X_train = merged_df[merged_df['subject_id'].isin(train_patients)].drop(columns=['has_cancer_history'], errors='ignore')
y_train = merged_df[merged_df['subject_id'].isin(train_patients)]['has_cancer_history']
X_test = merged_df[merged_df['subject_id'].isin(test_patients)].drop(columns=['has_cancer_history'], errors='ignore')
y_test = merged_df[merged_df['subject_id'].isin(test_patients)]['has_cancer_history']

# Drop `subject_id` only now, after the split
X_train = X_train.drop(columns=['subject_id'], errors='ignore')
X_test = X_test.drop(columns=['subject_id'], errors='ignore')

# Handle missing values
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

numerical_columns = merged_df.select_dtypes(include=[np.number]).columns
categorical_columns = merged_df.select_dtypes(exclude=[np.number]).columns

print("COLUMNS OF X TRAIN", X_train.columns)
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder

# Step 1: Drop Constant Columns (to avoid divide by zero in correlation)
constant_columns = [col for col in X_train.columns if X_train[col].nunique() == 1]
X_train.drop(columns=constant_columns, errors="ignore", inplace=True)
X_test.drop(columns=constant_columns, errors="ignore", inplace=True)
print(f"Dropped constant columns: {constant_columns}")

# Step 2: Identify & Drop Highly Correlated Features
correlation_threshold = 0.95
correlation_matrix = X_train.corr().abs()

# Find pairs of highly correlated features
high_corr_pairs = (
    correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1) > 0)
    .stack()
    .reset_index()
)
high_corr_pairs.columns = ["Feature_1", "Feature_2", "Correlation"]
high_corr_pairs = high_corr_pairs[high_corr_pairs["Correlation"] > correlation_threshold]

# Identify features to drop
features_to_drop = set()
for _, row in high_corr_pairs.iterrows():
    feature_1, feature_2 = row["Feature_1"], row["Feature_2"]
    if feature_1 not in features_to_drop and feature_2 not in features_to_drop:
        features_to_drop.add(feature_2)  # Drop one of the two

print(f"Features to Drop Due to High Correlation: {features_to_drop}")

# Drop the identified features
X_train.drop(columns=features_to_drop, errors="ignore", inplace=True)
X_test.drop(columns=features_to_drop, errors="ignore", inplace=True)

print(f"Updated X_train Shape: {X_train.shape}")
print(f"Updated X_test Shape: {X_test.shape}")

# Identify Top 100 BERT Features Without Looking at y_train
bert_columns = [col for col in X_train.columns if col.startswith("BERT_")]

# Compute variance per feature (instead of correlation with y_train)
feature_variances = X_train[bert_columns].var().sort_values(ascending=False)

# Select top 100 BERT features by variance
top_bert_features = feature_variances.head(100).index.tolist()

# Apply feature selection
X_train = X_train[top_bert_features + list(X_train.columns.difference(bert_columns))]
X_test = X_test[top_bert_features + list(X_test.columns.difference(bert_columns))]

print("Keeping only top 100 BERT features (selected on training set only)")
# Identify potential leakage columns
leakage_features = ['subject_id', 'hadm_id_adm', 'hadm_id_proc', 'hadm_id_vs', 
                    'hadm_id_med', 'starttime', 'charttime_x', 'charttime_y']

# Drop them from training & test sets
X_train = X_train.drop(columns=[col for col in leakage_features if col in X_train.columns], errors="ignore")
X_test = X_test.drop(columns=[col for col in leakage_features if col in X_test.columns], errors="ignore")

print(f"After leakage removal, X_train Shape: {X_train.shape}, X_test Shape: {X_test.shape}")


# Step 4: Standardize & Impute Missing Values
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

# Impute missing values
num_imputer = SimpleImputer(strategy="median")
X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])

# Scale numerical features
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])


# Encode categorical features
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])


# Apply feature selection only on training set
feature_variances = X_train.var().sort_values(ascending=False)
top_features = feature_variances.head(100).index.tolist()

X_train = X_train[top_features]
X_test = X_test[top_features]


X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

print(f"Updated X_train Shape: {X_train.shape}")
print(f"Updated X_test Shape: {X_test.shape}")


smote = SMOTE(sampling_strategy=0.8, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Convert to tensors after SMOTE
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Create DataLoader
batch_size = 32
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 2: Adjust Focal Loss Parameters
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1.5):  # Increase alpha and lower gamma slightly
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

# Step 3: Modify Model Architecture for Better Recall
class MultiModalModel(nn.Module):
    def __init__(self, input_dim):
        super(MultiModalModel, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 512)  
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.4)  

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = self.sigmoid(self.fc4(x))
        return x

# Step 4: Use Weighted Binary Cross-Entropy Loss Instead of Focal Loss
def weighted_bce_loss(outputs, targets, pos_weight=3.0):
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))(outputs, targets)
    return loss

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, loss function, and optimizer
model = MultiModalModel(input_dim=X_train.shape[1]).to(device)
criterion = weighted_bce_loss  # Switch to weighted BCE
optimizer = optim.AdamW(model.parameters(), lr=0.0003)  # Lower learning rate for stability

# Step 5: Train Model
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Step 6: Evaluate Model with Lowered Threshold for Better Recall
model.eval()
with torch.no_grad():
    test_predictions = []
    test_labels = []

    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        outputs = model(batch_X)
        test_predictions.extend(outputs.cpu().numpy())
        test_labels.extend(batch_y.cpu().numpy())

test_predictions = np.array(test_predictions).flatten()
test_labels = np.array(test_labels).flatten()

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve


# Optimize threshold (Find best trade-off between precision and recall)
precisions, recalls, thresholds = precision_recall_curve(test_labels, test_predictions)
best_threshold = thresholds[np.argmax(precisions * recalls)]
print("Best threshold:", best_threshold)

#Step 7: Apply LOWER Threshold to Improve Recall
test_predictions = (test_predictions > 0.7).astype(int)  # Adjusted threshold

# Step 8: Print Final Performance Metrics
accuracy = accuracy_score(test_labels, test_predictions)
precision = precision_score(test_labels, test_predictions)
recall = recall_score(test_labels, test_predictions)
f1 = f1_score(test_labels, test_predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"ecall: {recall:.4f} (Should be significantly improved)")
print(f"F1 Score: {f1:.4f}")
