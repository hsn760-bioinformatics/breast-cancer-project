"""
07_predict.py

Purpose:
- Load the saved DNN, scaler, gene list and class mapping
- Predict the PAM50 + claudin-low subtype for new samples
- Save predictions (label, confidence, per-class probabilities) to
  results/predictions/

Two ways to run it:
  1) DEMO (no arguments): predicts the first 10 held-out TEST samples and compares
     with their true labels, to prove the saved pipeline reproduces itself.
        python src/07_predict.py
  2) YOUR OWN FILE: a CSV with one row per sample and one column per gene.
        python src/07_predict.py data/new_samples/my_samples.csv

Input file rules:
- It must contain a column for EVERY gene in models/selected_feature_list.json
  (same lower-case gene names as the METABRIC file). Extra columns are ignored.
- Values must be on the same scale as the METABRIC expression data used for training
  (the model cannot correct for a different platform or normalisation).
- An optional 'patient_id' column is carried into the output.
"""

import os
import sys
import json

# Reduce TensorFlow's startup log noise (must be set BEFORE importing tensorflow)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("Starting prediction script...", flush=True)

import warnings
warnings.filterwarnings("ignore", message=".*structure of `inputs`.*")

import joblib
import numpy as np
import pandas as pd

print("Importing TensorFlow (can take 1-3 minutes on Windows, please wait, "
      "do NOT press Ctrl+C)...", flush=True)
import tensorflow as tf
print("TensorFlow imported.\n", flush=True)

# Predictions below this top-class probability are flagged. This is a simple
# heuristic, NOT a validated cut-off (softmax probabilities are not guaranteed
# to be calibrated).
LOW_CONFIDENCE_THRESHOLD = 0.60
DEMO_N = 10

# ---------------------------------------------------------------
# 1. Paths + existence checks
# ---------------------------------------------------------------
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
OUT_DIR = os.path.join("results", "predictions")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "breast_cancer_subtype_dnn.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
CLASS_MAP_PATH = os.path.join(MODELS_DIR, "class_mapping.json")
FEATURES_PATH = os.path.join(MODELS_DIR, "selected_feature_list.json")
X_TRAIN_PATH = os.path.join(PROCESSED_DIR, "X_train_selected.csv")

required = {
    MODEL_PATH: "04_train_model.py",
    SCALER_PATH: "04_train_model.py",
    CLASS_MAP_PATH: "02_preprocessing.py",
    FEATURES_PATH: "03_feature_selection.py",
    X_TRAIN_PATH: "03_feature_selection.py",
}
missing = [(p, s) for p, s in required.items() if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        "Missing required files:\n"
        + "\n".join(f"  {p}   (created by src/{s})" for p, s in missing)
        + "\nRe-run the listed script(s) in order (02 -> 03 -> 04), then try again."
    )

# ---------------------------------------------------------------
# 2. Load model + preprocessing objects
# ---------------------------------------------------------------
print("Loading model, scaler, gene list and class mapping...", flush=True)
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(CLASS_MAP_PATH) as f:
    class_mapping = {int(k): v for k, v in json.load(f).items()}
class_names = [class_mapping[i] for i in sorted(class_mapping)]

with open(FEATURES_PATH) as f:
    gene_names = json.load(f)

# Training-set medians (used only if a new sample has a missing gene value).
# Computed from the training split, exactly as preprocessing would have done.
train_medians = pd.read_csv(X_TRAIN_PATH)[gene_names].median()

print(f"Model expects {len(gene_names)} genes and predicts {len(class_names)} classes: "
      f"{class_names}\n")

# ---------------------------------------------------------------
# 3. Load the samples to predict
# ---------------------------------------------------------------
true_labels = None
if len(sys.argv) > 1:
    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file not found: '{input_path}'. Check the path (relative to the "
            f"project root) and the file name.")
    print(f"Reading samples from: {input_path}", flush=True)
    raw = pd.read_csv(input_path, low_memory=False)
    mode = "user_file"
else:
    demo_x = os.path.join(PROCESSED_DIR, "X_test_selected.csv")
    demo_y = os.path.join(PROCESSED_DIR, "y_test.csv")
    if not (os.path.exists(demo_x) and os.path.exists(demo_y)):
        raise FileNotFoundError(
            "Demo mode needs data/processed/X_test_selected.csv and y_test.csv "
            "(run 02 and 03 first), or pass your own CSV file as an argument.")
    print(f"No input file given -> DEMO MODE: first {DEMO_N} held-out test samples.",
          flush=True)
    raw = pd.read_csv(demo_x).head(DEMO_N)
    y_demo = pd.read_csv(demo_y)["target"].values[:DEMO_N]
    true_labels = [class_mapping[int(i)] for i in y_demo]
    mode = "demo"

print(f"Input table: {raw.shape[0]} samples x {raw.shape[1]} columns\n")
if raw.shape[0] == 0:
    raise ValueError("The input file has no rows.")

# ---------------------------------------------------------------
# 4. Validate genes and build the model input
# ---------------------------------------------------------------
missing_genes = [g for g in gene_names if g not in raw.columns]
if missing_genes:
    shown = ", ".join(missing_genes[:15]) + (" ..." if len(missing_genes) > 15 else "")
    raise ValueError(
        f"The input file is missing {len(missing_genes)} of the {len(gene_names)} genes "
        f"the model needs (e.g. {shown}). Gene column names must match "
        f"models/selected_feature_list.json exactly. The model cannot predict without "
        f"all of them.")

X_new = raw[gene_names].apply(pd.to_numeric, errors="coerce")   # non-numeric -> NaN
X_new = X_new.replace([np.inf, -np.inf], np.nan)

n_missing_cells = int(X_new.isnull().sum().sum())
if n_missing_cells > 0:
    frac = n_missing_cells / X_new.size
    print(f"NOTE: {n_missing_cells} missing/non-numeric values ({frac:.2%} of cells) "
          f"replaced with TRAINING-set gene medians.")
    if frac > 0.05:
        print("WARNING: more than 5% of values are missing; predictions may be unreliable.")
    X_new = X_new.fillna(train_medians)
    print()

# Same scaling as training (train-fitted scaler, transform only)
X_new_scaled = scaler.transform(X_new.values)

# Scale sanity check: values far outside the training range suggest a different
# platform/normalisation, which this model cannot handle.
z_extreme = np.abs(X_new_scaled) > 6
frac_extreme = z_extreme.mean()
if frac_extreme > 0.01:
    print(f"WARNING: {frac_extreme:.1%} of scaled values are more than 6 standard "
          f"deviations from the training mean. The input may use a different "
          f"platform or normalisation than the training data; treat predictions "
          f"with great caution.\n")

# ---------------------------------------------------------------
# 5. Predict
# ---------------------------------------------------------------
print("Predicting...", flush=True)
proba = model.predict(X_new_scaled, verbose=0)
top1 = np.argmax(proba, axis=1)
order = np.argsort(-proba, axis=1)
top2 = order[:, 1]

out = pd.DataFrame()
if "patient_id" in raw.columns:
    out["patient_id"] = raw["patient_id"].values
out["sample_index"] = np.arange(len(raw))
out["predicted_subtype"] = [class_mapping[i] for i in top1]
out["confidence"] = proba[np.arange(len(proba)), top1].round(4)
out["second_choice"] = [class_mapping[i] for i in top2]
out["second_choice_prob"] = proba[np.arange(len(proba)), top2].round(4)
out["low_confidence_flag"] = out["confidence"] < LOW_CONFIDENCE_THRESHOLD
# Per-sample check: many genes far outside the training range => the sample is
# probably on a different scale/platform, and its confidence is NOT trustworthy.
out["out_of_range_flag"] = z_extreme.mean(axis=1) > 0.05
for j, cname in enumerate(class_names):
    out[f"proba_{cname}"] = proba[:, j].round(4)

if true_labels is not None:
    out["true_subtype"] = true_labels
    out["correct"] = out["predicted_subtype"] == out["true_subtype"]

# ---------------------------------------------------------------
# 6. Report and save
# ---------------------------------------------------------------
print("=" * 70)
print("PREDICTIONS")
print("=" * 70)
show_cols = [c for c in ["patient_id", "sample_index", "predicted_subtype", "confidence",
                         "second_choice", "second_choice_prob", "low_confidence_flag",
                         "out_of_range_flag", "true_subtype", "correct"] if c in out.columns]
print(out[show_cols].to_string(index=False))
print()

n_low = int(out["low_confidence_flag"].sum())
print(f"Samples flagged low-confidence (top probability < {LOW_CONFIDENCE_THRESHOLD}): "
      f"{n_low} / {len(out)}")
n_oor = int(out["out_of_range_flag"].sum())
print(f"Samples flagged out-of-range (>5% of genes beyond 6 SD of training data): "
      f"{n_oor} / {len(out)}")
if n_oor > 0:
    print("  Out-of-range samples get unreliable predictions, however high the "
          "reported confidence is.")
print("Predicted subtype counts:")
print(out["predicted_subtype"].value_counts().to_string())
print()

if true_labels is not None:
    n_ok = int(out["correct"].sum())
    print(f"DEMO check: {n_ok}/{len(out)} correct on these {len(out)} held-out samples.")
    print("(Only a smoke test of the saved pipeline. Ten samples say nothing about "
          "overall accuracy; see results/metrics/test_metrics.json for that.)\n")

out_name = "predictions_demo.csv" if mode == "demo" else "predictions.csv"
out_path = os.path.join(OUT_DIR, out_name)
out.to_csv(out_path, index=False)

print("=" * 70)
print(f"Predictions saved to: {out_path}")
print("Research use only: this model is a thesis prototype and must not be used "
      "for clinical decisions.")
print("PREDICTION COMPLETE.")
print("=" * 70)