"""
06_explain_model.py

Purpose:
- Explain the trained DNN with SHAP (SHapley Additive exPlanations)
- Identify the genes that contribute most to the subtype predictions
  (globally and for each PAM50 subtype separately)
- Save figures to results/figures and tables to results/explainability

Method notes (for the thesis):
- Explainer: shap.DeepExplainer (exact DeepLIFT/Shapley-style attributions for
  neural networks). If it fails on your TensorFlow/SHAP versions the script falls
  back to shap.GradientExplainer (approximate) and says so.
- Background (reference) distribution: random TRAINING samples. Test samples are
  explained. Explaining does not change the model, so this is not test-set tuning.
- SHAP values are in units of predicted probability for each class, computed on the
  scaled gene-expression values that the model actually sees.
- SHAP describes what the MODEL uses. It does NOT prove a gene is causally or
  biologically important for the subtype.
"""

import os
import json

# Reduce TensorFlow's startup log noise (must be set BEFORE importing tensorflow)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("Starting SHAP explainability script...", flush=True)

import warnings
warnings.filterwarnings("ignore", message=".*structure of `inputs`.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*graph support has been removed.*")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

print("Importing TensorFlow and SHAP (can take 1-3 minutes on Windows, please wait, "
      "do NOT press Ctrl+C)...", flush=True)
import tensorflow as tf
import shap
print(f"Imported. TensorFlow {tf.__version__}, SHAP {shap.__version__}\n", flush=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

N_BACKGROUND = 100   # training samples used as the SHAP reference set (DeepExplainer uses at most 100)
TOP_N = 20           # number of top genes shown in plots / tables

# ---------------------------------------------------------------
# 1. Paths + existence checks
# ---------------------------------------------------------------
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
FIGURES_DIR = os.path.join("results", "figures")
EXPL_DIR = os.path.join("results", "explainability")
REPORTS_DIR = "reports"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(EXPL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "breast_cancer_subtype_dnn.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
CLASS_MAP_PATH = os.path.join(MODELS_DIR, "class_mapping.json")
FEATURES_PATH = os.path.join(MODELS_DIR, "selected_feature_list.json")
ANOVA_PATH = os.path.join(REPORTS_DIR, "feature_selection_anova_results.csv")

required = {
    MODEL_PATH: "04_train_model.py",
    SCALER_PATH: "04_train_model.py",
    CLASS_MAP_PATH: "02_preprocessing.py",
    FEATURES_PATH: "03_feature_selection.py",
    os.path.join(PROCESSED_DIR, "X_train_selected.csv"): "03_feature_selection.py",
    os.path.join(PROCESSED_DIR, "X_test_selected.csv"): "03_feature_selection.py",
    os.path.join(PROCESSED_DIR, "y_test.csv"): "02_preprocessing.py",
}
missing = [(p, s) for p, s in required.items() if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        "Missing required files:\n"
        + "\n".join(f"  {p}   (created by src/{s})" for p, s in missing)
        + "\nRe-run the listed script(s) in order (02 -> 03 -> 04), then try again."
    )

# ---------------------------------------------------------------
# 2. Load model, scaler, names and data
# ---------------------------------------------------------------
print("Loading model, scaler and data...", flush=True)
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(CLASS_MAP_PATH) as f:
    class_mapping = {int(k): v for k, v in json.load(f).items()}
class_ids = sorted(class_mapping.keys())
class_names = [class_mapping[i] for i in class_ids]
n_classes = len(class_ids)

with open(FEATURES_PATH) as f:
    gene_names = json.load(f)
n_features = len(gene_names)

X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_selected.csv"))
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_selected.csv"))
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv"))["target"].values

if X_train.columns.tolist() != gene_names or X_test.columns.tolist() != gene_names:
    raise ValueError("Feature columns do not match selected_feature_list.json. "
                     "Re-run 03 and 04.")

# Same scaling as training (train-fitted scaler, transform only)
X_train_s = scaler.transform(X_train.values)
X_test_s = scaler.transform(X_test.values)
n_test = X_test_s.shape[0]
print(f"Background pool (train): {X_train_s.shape}, samples to explain (test): "
      f"{X_test_s.shape}, classes: {class_names}\n", flush=True)

proba_test = model.predict(X_test_s, verbose=0)
pred_test = np.argmax(proba_test, axis=1)

# ---------------------------------------------------------------
# 3. SHAP computation
# ---------------------------------------------------------------
def pick_background(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    idx = rng.choice(X_train_s.shape[0], size=min(N_BACKGROUND, X_train_s.shape[0]),
                     replace=False)
    return X_train_s[idx]


def to_3d(sv, n_samples: int) -> np.ndarray:
    """Return SHAP values as an array (samples, features, classes) for any SHAP version."""
    if isinstance(sv, list):                     # older SHAP: list of (samples, features)
        arr = np.stack(sv, axis=-1)
    else:
        arr = np.asarray(sv)
    if arr.ndim != 3 or arr.shape != (n_samples, n_features, n_classes):
        raise ValueError(f"Unexpected SHAP output shape {arr.shape}; expected "
                         f"({n_samples}, {n_features}, {n_classes}).")
    return arr


def compute_shap(background: np.ndarray, data: np.ndarray):
    """Try DeepExplainer first (exact for this network); fall back to GradientExplainer."""
    try:
        explainer = shap.DeepExplainer(model, background)
        sv = explainer.shap_values(data, check_additivity=False)
        used = "DeepExplainer"
        expected = np.array(explainer.expected_value, dtype=float).ravel()
    except Exception as e:
        print(f"  DeepExplainer failed ({type(e).__name__}: {str(e)[:150]}). "
              f"Falling back to GradientExplainer (approximate).", flush=True)
        explainer = shap.GradientExplainer(model, background)
        sv = explainer.shap_values(data)
        used = "GradientExplainer"
        expected = model.predict(background, verbose=0).mean(axis=0)
    return to_3d(sv, data.shape[0]), expected, used


print("Computing SHAP values for the test set...", flush=True)
background = pick_background(RANDOM_SEED)
sv, expected_value, explainer_used = compute_shap(background, X_test_s)
print(f"  Explainer used: {explainer_used}; SHAP array shape: {sv.shape} "
      f"(samples, genes, classes)\n", flush=True)

# Sanity check: SHAP values + expected value should reproduce the model output
reconstructed = sv.sum(axis=1) + expected_value
add_err = np.abs(reconstructed - proba_test)
print("ADDITIVITY CHECK (SHAP sum + base value vs model probability)")
print(f"  max abs error: {add_err.max():.5f}, mean abs error: {add_err.mean():.5f}")
if add_err.max() > 0.05:
    print("  WARNING: error is not negligible. Treat rankings as approximate and "
          "mention this in the thesis.\n")
else:
    print("  OK: attributions reproduce the model output.\n")

# ---------------------------------------------------------------
# 4. Importance tables
# ---------------------------------------------------------------
abs_sv = np.abs(sv)
per_class_imp = abs_sv.mean(axis=0)              # (genes, classes): mean |SHAP| per class
global_imp = per_class_imp.sum(axis=1)           # summed over classes

per_class_df = pd.DataFrame(per_class_imp, index=gene_names, columns=class_names)
per_class_df.index.name = "gene"
global_df = pd.DataFrame({"gene": gene_names, "mean_abs_shap_all_classes": global_imp})
global_df = global_df.sort_values("mean_abs_shap_all_classes", ascending=False)
global_df["rank"] = np.arange(1, len(global_df) + 1)
global_df = global_df.merge(per_class_df.reset_index(), on="gene")
global_df.to_csv(os.path.join(EXPL_DIR, "global_gene_importance.csv"), index=False)

# Direction: does higher expression push the prediction TOWARD this class?
# Spearman correlation between the (scaled) gene value and its SHAP value for the class.
rows = []
for c, cname in enumerate(class_names):
    order = np.argsort(-per_class_imp[:, c])[:TOP_N]
    for rank, g in enumerate(order, start=1):
        shap_col = sv[:, g, c]
        if np.allclose(shap_col, shap_col[0]):
            rho = np.nan
        else:
            rho = spearmanr(X_test_s[:, g], shap_col)[0]
        if np.isnan(rho):
            direction = "undetermined"
        elif rho > 0.2:
            direction = "higher expression -> more likely this subtype"
        elif rho < -0.2:
            direction = "lower expression -> more likely this subtype"
        else:
            direction = "no clear monotonic direction"
        rows.append({
            "class": cname, "rank": rank, "gene": gene_names[g],
            "mean_abs_shap": per_class_imp[g, c],
            "spearman_expr_vs_shap": rho,
            "model_association": direction,
        })
top_per_class_df = pd.DataFrame(rows)
top_per_class_df.to_csv(os.path.join(EXPL_DIR, "top_genes_per_class.csv"), index=False)

print(f"TOP {TOP_N} GENES OVERALL (mean |SHAP| summed over the {n_classes} subtypes)")
print(global_df.head(TOP_N)[["rank", "gene", "mean_abs_shap_all_classes"]]
      .round(4).to_string(index=False))
print()
for cname in class_names:
    sub = top_per_class_df[top_per_class_df["class"] == cname].head(8)
    print(f"Top 8 genes for {cname}:")
    print(sub[["rank", "gene", "mean_abs_shap", "model_association"]]
          .round(4).to_string(index=False))
    print()

# ---------------------------------------------------------------
# 5. Robustness check: does the ranking depend on the background sample?
# ---------------------------------------------------------------
print("ROBUSTNESS CHECK (different random background sample)...", flush=True)
sv2, _, _ = compute_shap(pick_background(RANDOM_SEED + 1), X_test_s)
global_imp2 = np.abs(sv2).mean(axis=0).sum(axis=1)
rho_bg = spearmanr(global_imp, global_imp2)[0]
top_a = set(np.argsort(-global_imp)[:TOP_N])
top_b = set(np.argsort(-global_imp2)[:TOP_N])
overlap = len(top_a & top_b)
print(f"  Spearman correlation of gene importance across backgrounds (all genes): {rho_bg:.3f}")
print("  (The all-gene value is dragged down by the many near-zero, noise-level genes;")
print("   the top-gene overlap below is the more meaningful stability measure.)")
print(f"  Overlap of top-{TOP_N} genes: {overlap}/{TOP_N}\n", flush=True)

# ---------------------------------------------------------------
# 6. Comparison with the ANOVA feature-selection ranking (if available)
# ---------------------------------------------------------------
rho_anova = None
if os.path.exists(ANOVA_PATH):
    anova = pd.read_csv(ANOVA_PATH).set_index("gene")
    common = [g for g in gene_names if g in anova.index]
    rho_anova = spearmanr(global_df.set_index("gene").loc[common, "mean_abs_shap_all_classes"],
                          anova.loc[common, "f_score"])[0]
    top_anova = set(anova.loc[common].sort_values("f_score", ascending=False).head(TOP_N).index)
    top_shap = set(global_df.head(TOP_N)["gene"])
    print("COMPARISON WITH ANOVA F-SCORE RANKING (training set)")
    print(f"  Spearman(SHAP importance, ANOVA F-score): {rho_anova:.3f}")
    print(f"  Overlap of top-{TOP_N} (SHAP vs ANOVA): {len(top_anova & top_shap)}/{TOP_N}")
    print("  (Differences are expected: ANOVA is univariate, SHAP reflects what the "
          "multivariate model uses.)\n")
else:
    print("ANOVA results file not found; skipping SHAP vs ANOVA comparison.\n")

# ---------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------
print("Creating figures...", flush=True)
colors = plt.cm.tab10(np.arange(n_classes))

# 7a. Global bar plot: top genes, bar split by subtype contribution
top_idx = np.argsort(-global_imp)[:TOP_N][::-1]       # reversed so rank 1 is on top
fig, ax = plt.subplots(figsize=(9, 8))
left = np.zeros(len(top_idx))
for c, cname in enumerate(class_names):
    vals = per_class_imp[top_idx, c]
    ax.barh(range(len(top_idx)), vals, left=left, color=colors[c], label=cname)
    left += vals
ax.set_yticks(range(len(top_idx)))
ax.set_yticklabels([gene_names[i] for i in top_idx])
ax.set_xlabel("Mean |SHAP value| (summed over subtypes; colour = contribution per subtype)")
ax.set_title(f"Global feature importance: top {TOP_N} genes (test set)")
ax.legend(loc="lower right", title="Subtype")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "shap_bar_global_top20.png"), dpi=150)
plt.close()

# 7b. Per-class bar plots (2 x 3 grid)
ncols = 3
nrows = int(np.ceil(n_classes / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
axes = np.array(axes).ravel()
for c, cname in enumerate(class_names):
    order = np.argsort(-per_class_imp[:, c])[:15][::-1]
    axes[c].barh(range(len(order)), per_class_imp[order, c], color=colors[c])
    axes[c].set_yticks(range(len(order)))
    axes[c].set_yticklabels([gene_names[i] for i in order], fontsize=8)
    axes[c].set_title(f"{cname}: top 15 genes")
    axes[c].set_xlabel("Mean |SHAP value|")
for k in range(n_classes, len(axes)):
    axes[k].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "shap_bar_per_class_top15.png"), dpi=150)
plt.close()

# 7c. Beeswarm summary plot per subtype (shows direction and spread of effects)
for c, cname in enumerate(class_names):
    plt.figure()
    shap.summary_plot(sv[:, :, c], X_test_s, feature_names=gene_names,
                      max_display=TOP_N, show=False)
    plt.title(f"SHAP summary: {cname} (colour = scaled gene expression)")
    plt.tight_layout()
    safe = cname.replace("/", "_").replace(" ", "_")
    plt.savefig(os.path.join(FIGURES_DIR, f"shap_summary_{safe}.png"),
                dpi=150, bbox_inches="tight")
    plt.close("all")

# 7d. Heatmap of per-class importance for the overall top genes
top_heat = np.argsort(-global_imp)[:TOP_N]
plt.figure(figsize=(8, 8))
plt.imshow(per_class_imp[top_heat], aspect="auto", cmap="viridis")
plt.colorbar(label="Mean |SHAP value|")
plt.xticks(range(n_classes), class_names, rotation=45, ha="right")
plt.yticks(range(len(top_heat)), [gene_names[i] for i in top_heat])
plt.title(f"Which subtype does each top gene help predict? (top {TOP_N})")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "shap_heatmap_top_genes_by_class.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------
# 8. Save raw SHAP values + a plain-text summary with limitations
# ---------------------------------------------------------------
np.savez_compressed(
    os.path.join(EXPL_DIR, "shap_values_test.npz"),
    shap_values=sv.astype(np.float32),
    expected_value=expected_value,
    gene_names=np.array(gene_names),
    class_names=np.array(class_names),
    y_true=y_test, y_pred=pred_test,
)

summary_lines = [
    "SHAP EXPLAINABILITY SUMMARY",
    "=" * 60,
    f"Explainer: {explainer_used}",
    f"Samples explained (test set): {n_test}; background samples (train): {len(background)}",
    f"Additivity error: max {add_err.max():.5f}, mean {add_err.mean():.5f}",
    f"Background robustness: Spearman {rho_bg:.3f}, top-{TOP_N} overlap {overlap}/{TOP_N}",
]
if rho_anova is not None:
    summary_lines.append(f"Spearman(SHAP importance, ANOVA F-score): {rho_anova:.3f}")
summary_lines += [
    "",
    f"Top {TOP_N} genes overall: " + ", ".join(global_df.head(TOP_N)["gene"].tolist()),
    "",
    "LIMITATIONS OF THIS ANALYSIS",
    "- SHAP shows what the trained model relies on, not causal biology.",
    "- Gene names are taken from the dataset's column headers (lower-case symbols). "
    "They were not independently verified against a gene-annotation database.",
    "- Correlated genes share credit, so a gene's SHAP importance can be split with "
    "correlated neighbours.",
    "- Values depend on the background (reference) sample; see robustness check above.",
    "- Explained samples are the test set (n=" + str(n_test) + "); rankings for small "
    "classes rest on few samples.",
    "- Any biological interpretation must be supported by the literature, not by "
    "SHAP alone.",
]
with open(os.path.join(EXPL_DIR, "shap_summary.txt"), "w") as f:
    f.write("\n".join(summary_lines))

print("=" * 70)
print("SAVED FILES")
print("=" * 70)
print(f"  {EXPL_DIR}/global_gene_importance.csv, top_genes_per_class.csv,")
print(f"  {EXPL_DIR}/shap_values_test.npz, shap_summary.txt")
print(f"  {FIGURES_DIR}/shap_bar_global_top20.png, shap_bar_per_class_top15.png,")
print(f"  {FIGURES_DIR}/shap_heatmap_top_genes_by_class.png, shap_summary_<subtype>.png (x{n_classes})")
print("\nEXPLAINABILITY COMPLETE. Next step: 07_predict.py")
print("=" * 70)