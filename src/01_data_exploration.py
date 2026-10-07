"""
01_data_exploration.py

Purpose: Inspect the raw METABRIC dataset structure BEFORE making any
assumptions about which columns are gene-expression features, clinical
variables, mutation variables, or the classification target.

This script does NOT modify or clean data. It only reports facts about
the dataset so that later scripts can be written correctly.
"""

import os
import pandas as pd

# ---------------------------------------------------------------
# 1. Paths (relative to project root, so this works regardless of
#    where VS Code's terminal cwd happens to be, as long as you run
#    it from the project root as instructed below)
# ---------------------------------------------------------------
RAW_DATA_PATH = os.path.join("data", "raw", "METABRIC_RNA_Mutation.csv")
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------
if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(
        f"Could not find dataset at '{RAW_DATA_PATH}'. "
        f"Make sure you are running this command from the project root "
        f"(Breast_Cancer_Project\\) and that the CSV is in data/raw/."
    )

print(f"Loading dataset from: {RAW_DATA_PATH}")
df = pd.read_csv(RAW_DATA_PATH)
print(f"Dataset loaded successfully. Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

# ---------------------------------------------------------------
# 3. Basic structure
# ---------------------------------------------------------------
print("=" * 70)
print("BASIC STRUCTURE")
print("=" * 70)
print(df.dtypes.value_counts())
print()

# ---------------------------------------------------------------
# 4. List ALL column names with their dtype and number of missing values
#    This is the single most important output of this script.
# ---------------------------------------------------------------
col_summary = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.astype(str).values,
    "n_missing": df.isnull().sum().values,
    "pct_missing": (df.isnull().sum().values / len(df) * 100).round(2),
    "n_unique": [df[c].nunique(dropna=True) for c in df.columns],
})

col_summary_path = os.path.join(REPORTS_DIR, "column_summary_full.csv")
col_summary.to_csv(col_summary_path, index=False)
print(f"Full column summary (all {df.shape[1]} columns) saved to: {col_summary_path}")
print("Open this CSV in Excel/VS Code and inspect it carefully.\n")

# ---------------------------------------------------------------
# 5. Candidate identification: mutation columns
#    METABRIC mutation columns conventionally end in '_mut'
# ---------------------------------------------------------------
mutation_cols = [c for c in df.columns if c.lower().endswith("_mut")]
print("=" * 70)
print(f"CANDIDATE MUTATION COLUMNS (suffix '_mut'): {len(mutation_cols)} found")
print("=" * 70)
print(mutation_cols[:20], "..." if len(mutation_cols) > 20 else "")
print()

# ---------------------------------------------------------------
# 6. Candidate identification: gene expression columns
#    In METABRIC_RNA_Mutation.csv, gene expression z-scores are
#    typically float64 columns that are NOT mutation columns and NOT
#    known clinical/survival/treatment fields. We identify them by
#    ELIMINATION and print them for manual verification -- we do NOT
#    assume this list is correct without you checking it.
# ---------------------------------------------------------------

# Known non-gene-expression columns commonly present in this dataset.
# This list is a STARTING POINT based on typical METABRIC column names,
# not a hard assumption -- we verify against your actual columns below.
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

candidate_gene_cols = [
    c for c in df.columns
    if df[c].dtype == "float64"
    and not c.lower().endswith("_mut")
    and not looks_like_known_clinical(c)
]

print("=" * 70)
print(f"CANDIDATE GENE-EXPRESSION COLUMNS (float64, not mutation, not known clinical): {len(candidate_gene_cols)} found")
print("=" * 70)
print("First 20 candidates:")
print(candidate_gene_cols[:20])
print("Last 20 candidates:")
print(candidate_gene_cols[-20:])
print()

candidate_path = os.path.join(REPORTS_DIR, "candidate_gene_expression_columns.csv")
pd.Series(candidate_gene_cols, name="candidate_gene_column").to_csv(candidate_path, index=False)
print(f"Full candidate list saved to: {candidate_path}")
print("STOP: Open this file and confirm these are gene expression columns")
print("(they should look like gene symbols, e.g. 'brca1', 'tp53', 'erbb2')\n")

# ---------------------------------------------------------------
# 7. Candidate identification: target column
# ---------------------------------------------------------------
print("=" * 70)
print("TARGET COLUMN CANDIDATES")
print("=" * 70)
target_keywords = ["subtype", "pam50", "cluster", "class"]
target_candidates = [c for c in df.columns if any(k in c.lower() for k in target_keywords)]
print("Columns matching subtype/class-like keywords:")
print(target_candidates)
print()

for col in target_candidates:
    print(f"--- {col} ---")
    print(f"  dtype: {df[col].dtype}")
    print(f"  missing values: {df[col].isnull().sum()}")
    print(f"  unique values: {df[col].dropna().unique()}")
    print(f"  value counts:\n{df[col].value_counts(dropna=False)}")
    print()

# ---------------------------------------------------------------
# 8. Object/string columns overview (helps identify clinical vs other)
# ---------------------------------------------------------------
object_cols = df.select_dtypes(include="object").columns.tolist()
print("=" * 70)
print(f"OBJECT/STRING COLUMNS: {len(object_cols)}")
print("=" * 70)
print(object_cols)
print()

print("=" * 70)
print("INSPECTION COMPLETE.")
print("Next step: review reports/column_summary_full.csv and")
print("reports/candidate_gene_expression_columns.csv before we write")
print("02_preprocessing.py. Do not proceed until these look correct.")
print("=" * 70)