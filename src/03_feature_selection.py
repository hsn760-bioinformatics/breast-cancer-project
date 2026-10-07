"""
03_feature_selection.py

Purpose:
- Load the preprocessed train/val/test splits (gene expression only)
- Perform STATISTICAL feature selection: one-way ANOVA F-test per gene,
  computed on the TRAINING SET ONLY, with Benjamini-Hochberg FDR
  correction across all tests (since testing ~489 genes simultaneously
  at raw p<0.05 would yield a substantial number of false positives by
  chance alone).
- Apply the resulting gene list to train/val/test (val/test never
  influence which genes are selected -- no leakage).
- Save the reduced feature matrices and the selected gene list.

This is explicitly a STATISTICAL feature selection step, distinct from:
- biological filtering (already done in 02: restricting to gene-
  expression columns only, excluding clinical/mutation variables)
- model-based feature selection (not performed in this pipeline;
  would mean using e.g. a trained model's importances to select genes)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

FDR_ALPHA = 0.05  # target false discovery rate

# ---------------------------------------------------------------
# 1. Load processed data
# ---------------------------------------------------------------
print("Loading processed train/val/test splits...")
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
X_val = pd.read_csv(os.path.join(PROCESSED_DIR, "X_val.csv"))
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))

y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv"))["target"].values
y_val = pd.read_csv(os.path.join(PROCESSED_DIR, "y_val.csv"))["target"].values
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv"))["target"].values

print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}\n")

gene_names = X_train.columns.tolist()

# ---------------------------------------------------------------
# 2. ANOVA F-test per gene, TRAIN ONLY
# ---------------------------------------------------------------
print("Running ANOVA F-test per gene (training set only)...")
f_scores, p_values = f_classif(X_train.values, y_train)

results_df = pd.DataFrame({
    "gene": gene_names,
    "f_score": f_scores,
    "p_value": p_values,
})

# ---------------------------------------------------------------
# 3. Benjamini-Hochberg FDR correction (manual implementation,
#    no extra dependency required)
# ---------------------------------------------------------------
def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05):
    """
    Returns adjusted p-values (q-values) and a boolean mask of which
    hypotheses are rejected (i.e. considered significant) at the given
    false discovery rate.
    """
    n = len(p_values)
    order = np.argsort(p_values)
    ranked_p = p_values[order]

    # BH adjusted p-values
    adjusted = ranked_p * n / (np.arange(1, n + 1))
    # enforce monotonicity (standard step-up correction)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    # map back to original order
    q_values = np.empty(n)
    q_values[order] = adjusted

    reject = q_values <= alpha
    return q_values, reject

q_values, reject_mask = benjamini_hochberg(results_df["p_value"].values, FDR_ALPHA)
results_df["q_value"] = q_values
results_df["significant"] = reject_mask

results_df = results_df.sort_values("p_value").reset_index(drop=True)

# ---------------------------------------------------------------
# 4. Report and save full statistical results (for the thesis appendix)
# ---------------------------------------------------------------
n_significant = int(results_df["significant"].sum())
print("=" * 70)
print("ANOVA F-TEST RESULTS (with Benjamini-Hochberg FDR correction)")
print("=" * 70)
print(f"Total genes tested: {len(results_df)}")
print(f"Genes significant at FDR < {FDR_ALPHA}: {n_significant}")
print()
print("Top 15 most significant genes:")
print(results_df.head(15)[["gene", "f_score", "p_value", "q_value"]].to_string(index=False))
print()

full_results_path = os.path.join(REPORTS_DIR, "feature_selection_anova_results.csv")
results_df.to_csv(full_results_path, index=False)
print(f"Full ANOVA results (all {len(results_df)} genes) saved to: {full_results_path}\n")

# ---------------------------------------------------------------
# 5. Safety check: if the FDR filter is too aggressive or too lax,
#    fall back sensibly rather than silently producing a degenerate
#    feature set.
# ---------------------------------------------------------------
MIN_FEATURES = 20  # a DNN with fewer than this is unlikely to be meaningful here

if n_significant < MIN_FEATURES:
    print(f"WARNING: only {n_significant} genes passed FDR < {FDR_ALPHA}, "
          f"which is below the minimum of {MIN_FEATURES}. "
          f"Falling back to the top {MIN_FEATURES} genes by raw p-value instead. "
          f"This fallback should be reported as a limitation in the thesis.")
    selected_genes = results_df.head(MIN_FEATURES)["gene"].tolist()
else:
    selected_genes = results_df[results_df["significant"]]["gene"].tolist()

print(f"Final number of selected genes: {len(selected_genes)}\n")

# ---------------------------------------------------------------
# 6. Apply selection to train/val/test (selection decided on train only)
# ---------------------------------------------------------------
X_train_sel = X_train[selected_genes]
X_val_sel = X_val[selected_genes]
X_test_sel = X_test[selected_genes]

print(f"X_train_selected: {X_train_sel.shape}")
print(f"X_val_selected: {X_val_sel.shape}")
print(f"X_test_selected: {X_test_sel.shape}\n")

# ---------------------------------------------------------------
# 7. Save selected feature matrices and updated feature list
# ---------------------------------------------------------------
X_train_sel.to_csv(os.path.join(PROCESSED_DIR, "X_train_selected.csv"), index=False)
X_val_sel.to_csv(os.path.join(PROCESSED_DIR, "X_val_selected.csv"), index=False)
X_test_sel.to_csv(os.path.join(PROCESSED_DIR, "X_test_selected.csv"), index=False)

with open(os.path.join(MODELS_DIR, "selected_feature_list.json"), "w") as f:
    json.dump(selected_genes, f, indent=2)

print("=" * 70)
print("SAVED FILES")
print("=" * 70)
print(f"  {PROCESSED_DIR}/X_train_selected.csv, X_val_selected.csv, X_test_selected.csv")
print(f"  {MODELS_DIR}/selected_feature_list.json")
print(f"  {full_results_path}")
print()
print("FEATURE SELECTION COMPLETE. Next step: 04_train_model.py")
print("=" * 70)