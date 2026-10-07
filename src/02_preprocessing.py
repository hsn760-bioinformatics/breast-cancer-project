"""
02_preprocessing.py

Purpose:
- Load raw METABRIC data
- Define the target (PAM50 + claudin-low molecular subtype)
- Define the feature set (gene-expression columns ONLY)
- Clean target labels (drop missing / 'NC')
- Split into train/validation/test BEFORE learning any statistics
- Impute missing values and remove zero/near-zero variance features
  using train-set statistics only (no leakage from val/test)
- Save processed splits, label encoding, and feature list to disk
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------
RAW_DATA_PATH = os.path.join("data", "raw", "METABRIC_RNA_Mutation.csv")
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET_COL = "pam50_+_claudin-low_subtype"

# ---------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------
print(f"Loading dataset from: {RAW_DATA_PATH}")
df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
print(f"Loaded shape: {df.shape}\n")

# ---------------------------------------------------------------
# 3. Define feature columns: gene expression ONLY
#
#    We identify candidates the same way as script 01 (float64,
#    not a mutation column, not a known clinical/meta column), then
#    additionally remove 'neoplasm_histologic_grade', which script 01
#    flagged as a non-gene clinical variable that slipped through the
#    automatic filter.
# ---------------------------------------------------------------
known_clinical_or_meta_keywords = [
    "patient_id", "age_at_diagnosis", "type_of_breast_surgery",
    "cancer_type", "cancer_type_detailed", "cellularity",
    "chemotherapy", "pam50", "claudin", "er_status", "her2_status",
    "tumor_other_histologic", "hormone_therapy", "inferred_menopausal",
    "integrative_cluster", "primary_tumor_laterality", "lymph_nodes",
    "mutation_count", "nottingham", "oncotree", "overall_survival",
    "pr_status", "radio_therapy", "3-gene_classifier_subtype",
    "tumor_size", "tumor_stage", "death_from_cancer", "age_at",
    "cohort", "vital_status",
]

def looks_like_known_clinical(colname: str) -> bool:
    name_lower = colname.lower()
    return any(kw in name_lower for kw in known_clinical_or_meta_keywords)

EXPLICITLY_EXCLUDED_NON_GENE = {"neoplasm_histologic_grade"}

gene_cols = [
    c for c in df.columns
    if df[c].dtype == "float64"
    and not c.lower().endswith("_mut")
    and not looks_like_known_clinical(c)
    and c not in EXPLICITLY_EXCLUDED_NON_GENE
]

print(f"Identified {len(gene_cols)} gene-expression feature columns.")
if len(gene_cols) != 489:
    print(f"WARNING: expected 489 gene columns based on prior inspection, "
          f"found {len(gene_cols)}. Review the exclusion lists above before continuing.")
print()

# ---------------------------------------------------------------
# 4. Clean target
# ---------------------------------------------------------------
print("=" * 70)
print("TARGET CLEANING")
print("=" * 70)
print(f"Target column: {TARGET_COL}")
print("Class counts BEFORE cleaning:")
print(df[TARGET_COL].value_counts(dropna=False))
print()

# 'NC' means "not classified" -- not a real class, treat like missing.
df[TARGET_COL] = df[TARGET_COL].replace("NC", np.nan)

before_n = len(df)
df_clean = df.dropna(subset=[TARGET_COL]).copy()
after_n = len(df_clean)
print(f"Dropped {before_n - after_n} rows with missing/NC target "
      f"({before_n} -> {after_n} rows).")
print("Class counts AFTER cleaning:")
print(df_clean[TARGET_COL].value_counts())
print()

# ---------------------------------------------------------------
# 5. Encode target
# ---------------------------------------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df_clean[TARGET_COL])
class_mapping = {int(i): cls for i, cls in enumerate(label_encoder.classes_)}
print("Class encoding (integer -> label):")
for i, cls in class_mapping.items():
    print(f"  {i} -> {cls}")
print()

# ---------------------------------------------------------------
# 6. Build X (gene expression only) and y
# ---------------------------------------------------------------
X = df_clean[gene_cols].copy()
y = y_encoded

print(f"Feature matrix X shape: {X.shape}")
print(f"Target vector y shape: {y.shape}\n")

# ---------------------------------------------------------------
# 7. Check for infinite values and convert to NaN (so they can be
#    imputed the same way as missing values)
# ---------------------------------------------------------------
n_inf = np.isinf(X.values).sum()
print(f"Infinite values found in X: {n_inf}")
if n_inf > 0:
    X = X.replace([np.inf, -np.inf], np.nan)
    print("Infinite values replaced with NaN for imputation.\n")
else:
    print()

# ---------------------------------------------------------------
# 8. Check missing values in gene-expression features
# ---------------------------------------------------------------
missing_per_col = X.isnull().sum()
n_cols_with_missing = (missing_per_col > 0).sum()
print(f"Gene columns with at least one missing value: {n_cols_with_missing} / {X.shape[1]}")
print(f"Total missing cells in X: {missing_per_col.sum()}\n")

# ---------------------------------------------------------------
# 9. Train / validation / test split (STRATIFIED, done BEFORE any
#    statistic -- imputation medians, variance thresholds -- is
#    learned, to prevent data leakage from val/test into preprocessing)
#
#    Split: 70% train, 15% validation, 15% test
# ---------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_SEED
)

print("=" * 70)
print("SPLIT SIZES")
print("=" * 70)
print(f"Train: {X_train.shape[0]} samples")
print(f"Validation: {X_val.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples\n")

print("Class distribution (train):")
print(pd.Series(y_train).value_counts().sort_index())
print("\nClass distribution (validation):")
print(pd.Series(y_val).value_counts().sort_index())
print("\nClass distribution (test):")
print(pd.Series(y_test).value_counts().sort_index())
print()

# ---------------------------------------------------------------
# 10. Impute missing values using TRAIN-set medians only
# ---------------------------------------------------------------
train_medians = X_train.median()

X_train = X_train.fillna(train_medians)
X_val = X_val.fillna(train_medians)
X_test = X_test.fillna(train_medians)

# Any gene that is entirely missing in train would produce a NaN median;
# guard against that explicitly rather than silently propagating NaNs.
still_missing = X_train.isnull().sum()
genes_all_missing_in_train = still_missing[still_missing > 0].index.tolist()
if genes_all_missing_in_train:
    print(f"WARNING: {len(genes_all_missing_in_train)} genes had no valid "
          f"values in the training set and could not be imputed. "
          f"Dropping them from X_train/X_val/X_test: {genes_all_missing_in_train}")
    X_train = X_train.drop(columns=genes_all_missing_in_train)
    X_val = X_val.drop(columns=genes_all_missing_in_train)
    X_test = X_test.drop(columns=genes_all_missing_in_train)

print("Missing-value imputation complete (train-set medians applied to all splits).\n")

# ---------------------------------------------------------------
# 11. Remove zero-variance and near-zero-variance features
#     (computed on TRAIN ONLY, then the same columns are dropped
#     from val/test to keep feature sets identical)
#
#     Rationale: a gene with (near-)zero variance across training
#     patients carries no information the model can use to
#     discriminate between subtypes, and including it only adds
#     noise/dimensionality without predictive value.
# ---------------------------------------------------------------
VARIANCE_THRESHOLD = 0.01  # conservative: removes only near-constant genes

train_variances = X_train.var()
low_variance_genes = train_variances[train_variances <= VARIANCE_THRESHOLD].index.tolist()

print(f"Genes with variance <= {VARIANCE_THRESHOLD} in training set: {len(low_variance_genes)}")
if low_variance_genes:
    print(f"Removing: {low_variance_genes}")
    X_train = X_train.drop(columns=low_variance_genes)
    X_val = X_val.drop(columns=low_variance_genes)
    X_test = X_test.drop(columns=low_variance_genes)
print(f"Remaining gene features after variance filtering: {X_train.shape[1]}\n")

final_gene_cols = X_train.columns.tolist()

# ---------------------------------------------------------------
# 12. Save processed splits, label encoder, and feature list
# ---------------------------------------------------------------
X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
X_val.to_csv(os.path.join(PROCESSED_DIR, "X_val.csv"), index=False)
X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)

pd.Series(y_train, name="target").to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
pd.Series(y_val, name="target").to_csv(os.path.join(PROCESSED_DIR, "y_val.csv"), index=False)
pd.Series(y_test, name="target").to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.joblib"))

with open(os.path.join(MODELS_DIR, "class_mapping.json"), "w") as f:
    json.dump(class_mapping, f, indent=2)

with open(os.path.join(MODELS_DIR, "feature_list.json"), "w") as f:
    json.dump(final_gene_cols, f, indent=2)

print("=" * 70)
print("SAVED FILES")
print("=" * 70)
print(f"  {PROCESSED_DIR}/X_train.csv, X_val.csv, X_test.csv")
print(f"  {PROCESSED_DIR}/y_train.csv, y_val.csv, y_test.csv")
print(f"  {MODELS_DIR}/label_encoder.joblib")
print(f"  {MODELS_DIR}/class_mapping.json")
print(f"  {MODELS_DIR}/feature_list.json")
print()
print("PREPROCESSING COMPLETE. Next step: 03_feature_selection.py")
print("=" * 70)