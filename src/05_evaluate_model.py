"""
05_evaluate_model.py

Purpose:
- Load the trained DNN, the scaler (fit on train only) and the held-out TEST set
- Evaluate ONCE on the test set (no tuning is done on test data)
- Report accuracy, balanced accuracy, precision, recall, F1, MCC,
  classification report, confusion matrices, ROC-AUC (one-vs-rest) and ROC curves
- Report bootstrap 95% confidence intervals (test set is small: ~285 samples)
- Save all figures to results/figures and all numbers to results/metrics
"""

import os
import json

# Reduce TensorFlow's startup log noise (must be set BEFORE importing tensorflow)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("Starting evaluation script...", flush=True)

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, auc, matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize

print("Importing TensorFlow (can take 1-3 minutes on Windows, please wait, "
      "do NOT press Ctrl+C)...", flush=True)
import tensorflow as tf
print("TensorFlow imported.\n", flush=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ---------------------------------------------------------------
# 1. Paths + existence checks (clear errors instead of cryptic ones)
# ---------------------------------------------------------------
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
FIGURES_DIR = os.path.join("results", "figures")
METRICS_DIR = os.path.join("results", "metrics")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "breast_cancer_subtype_dnn.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
CLASS_MAP_PATH = os.path.join(MODELS_DIR, "class_mapping.json")
FEATURES_PATH = os.path.join(MODELS_DIR, "selected_feature_list.json")

required = {
    MODEL_PATH: "04_train_model.py",
    SCALER_PATH: "04_train_model.py",
    CLASS_MAP_PATH: "02_preprocessing.py",
    FEATURES_PATH: "03_feature_selection.py",
    os.path.join(PROCESSED_DIR, "X_test_selected.csv"): "03_feature_selection.py",
    os.path.join(PROCESSED_DIR, "y_test.csv"): "02_preprocessing.py",
}
missing = [(p, s) for p, s in required.items() if not os.path.exists(p)]
if missing:
    msg = "Missing required files:\n" + "\n".join(
        f"  {p}   (created by src/{s})" for p, s in missing
    ) + "\nRe-run the listed script(s) in order (02 -> 03 -> 04), then try again."
    raise FileNotFoundError(msg)

# ---------------------------------------------------------------
# 2. Load everything
# ---------------------------------------------------------------
print("Loading model, scaler, class mapping and test data...", flush=True)
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(CLASS_MAP_PATH) as f:
    class_mapping = {int(k): v for k, v in json.load(f).items()}
class_ids = sorted(class_mapping.keys())
class_names = [class_mapping[i] for i in class_ids]

with open(FEATURES_PATH) as f:
    selected_features = json.load(f)

X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_selected.csv"))
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv"))["target"].values

# The model expects features in exactly the training order
if X_test.columns.tolist() != selected_features:
    raise ValueError("Test feature columns do not match selected_feature_list.json "
                     "(different genes or different order). Re-run 03 and 04.")

print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} genes, "
      f"{len(class_ids)} classes: {class_names}\n")

# Scale with the TRAIN-fitted scaler (transform only, never fit on test)
X_test_scaled = scaler.transform(X_test.values)

# ---------------------------------------------------------------
# 3. Predict
# ---------------------------------------------------------------
print("Predicting on the test set...", flush=True)
proba = model.predict(X_test_scaled, verbose=0)
y_pred = np.argmax(proba, axis=1)

# ---------------------------------------------------------------
# 4. Overall metrics
# ---------------------------------------------------------------
acc = accuracy_score(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
    y_test, y_pred, average="macro", zero_division=0)
p_wt, r_wt, f_wt, _ = precision_recall_fscore_support(
    y_test, y_pred, average="weighted", zero_division=0)

# One-vs-rest ROC-AUC (appropriate for multiclass with softmax probabilities)
auc_macro = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
auc_weighted = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")

print("=" * 70)
print("TEST SET PERFORMANCE")
print("=" * 70)
print(f"Accuracy:            {acc:.4f}")
print(f"Balanced accuracy:   {bal_acc:.4f}")
print(f"MCC:                 {mcc:.4f}")
print(f"Precision (macro):   {p_macro:.4f}   (weighted: {p_wt:.4f})")
print(f"Recall    (macro):   {r_macro:.4f}   (weighted: {r_wt:.4f})")
print(f"F1        (macro):   {f_macro:.4f}   (weighted: {f_wt:.4f})")
print(f"ROC-AUC OvR (macro): {auc_macro:.4f}   (weighted: {auc_weighted:.4f})\n")

# ---------------------------------------------------------------
# 5. Bootstrap 95% confidence intervals (test set is small)
# ---------------------------------------------------------------
rng = np.random.default_rng(RANDOM_SEED)
n = len(y_test)
boot_acc, boot_f1 = [], []
for _ in range(1000):
    idx = rng.integers(0, n, n)
    boot_acc.append(accuracy_score(y_test[idx], y_pred[idx]))
    boot_f1.append(f1_score(y_test[idx], y_pred[idx], average="macro", zero_division=0))
acc_ci = np.percentile(boot_acc, [2.5, 97.5])
f1_ci = np.percentile(boot_f1, [2.5, 97.5])
print(f"Accuracy 95% bootstrap CI:  [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]")
print(f"Macro-F1 95% bootstrap CI:  [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]\n")

# ---------------------------------------------------------------
# 6. Classification report (per class)
# ---------------------------------------------------------------
report_text = classification_report(
    y_test, y_pred, labels=class_ids, target_names=class_names,
    digits=3, zero_division=0)
print("CLASSIFICATION REPORT")
print(report_text)

report_dict = classification_report(
    y_test, y_pred, labels=class_ids, target_names=class_names,
    output_dict=True, zero_division=0)

# Per-class one-vs-rest AUC + ROC curves
y_bin = label_binarize(y_test, classes=class_ids)
per_class_auc = {}
fig, ax = plt.subplots(figsize=(8, 7))
for i, name in zip(class_ids, class_names):
    fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
    per_class_auc[name] = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {per_class_auc[name]:.3f})")
ax.plot([0, 1], [0, 1], "k--", label="Chance")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("One-vs-rest ROC curves (test set)")
ax.legend(loc="lower right")
plt.tight_layout()
roc_path = os.path.join(FIGURES_DIR, "roc_curves_test.png")
plt.savefig(roc_path, dpi=150)
plt.close()

per_class_df = pd.DataFrame({
    "class": class_names,
    "precision": [report_dict[c]["precision"] for c in class_names],
    "recall": [report_dict[c]["recall"] for c in class_names],
    "f1": [report_dict[c]["f1-score"] for c in class_names],
    "support": [int(report_dict[c]["support"]) for c in class_names],
    "roc_auc_ovr": [per_class_auc[c] for c in class_names],
})
print("Per-class table (with one-vs-rest AUC):")
print(per_class_df.round(3).to_string(index=False))
print()

# ---------------------------------------------------------------
# 7. Confusion matrices (counts and row-normalised)
# ---------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred, labels=class_ids)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

for data, fmt, title, fname in [
    (cm, "d", "Confusion matrix (counts, test set)", "confusion_matrix_counts.png"),
    (cm_norm, ".2f", "Confusion matrix (row-normalised = recall, test set)",
     "confusion_matrix_normalized.png"),
]:
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted subtype")
    plt.ylabel("True subtype")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, fname), dpi=150)
    plt.close()

# Most frequent confusions, computed from the data (not assumed)
confusions = []
for i in range(len(class_ids)):
    for j in range(len(class_ids)):
        if i != j and cm[i, j] > 0:
            confusions.append((class_names[i], class_names[j], int(cm[i, j])))
confusions.sort(key=lambda t: t[2], reverse=True)
print("Most frequent misclassifications (true -> predicted):")
for t, p, c in confusions[:6]:
    print(f"  {t} -> {p}: {c} samples")
print()

# ---------------------------------------------------------------
# 8. Save numeric results
# ---------------------------------------------------------------
metrics = {
    "n_test_samples": int(n),
    "accuracy": acc,
    "accuracy_95ci": [float(acc_ci[0]), float(acc_ci[1])],
    "balanced_accuracy": bal_acc,
    "mcc": mcc,
    "precision_macro": p_macro, "recall_macro": r_macro, "f1_macro": f_macro,
    "f1_macro_95ci": [float(f1_ci[0]), float(f1_ci[1])],
    "precision_weighted": p_wt, "recall_weighted": r_wt, "f1_weighted": f_wt,
    "roc_auc_ovr_macro": auc_macro, "roc_auc_ovr_weighted": auc_weighted,
    "per_class_roc_auc": per_class_auc,
}
with open(os.path.join(METRICS_DIR, "test_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

with open(os.path.join(METRICS_DIR, "classification_report_test.txt"), "w") as f:
    f.write(report_text)

per_class_df.to_csv(os.path.join(METRICS_DIR, "per_class_metrics_test.csv"), index=False)
pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
    os.path.join(METRICS_DIR, "confusion_matrix_test.csv"))

pred_df = pd.DataFrame(proba, columns=[f"proba_{c}" for c in class_names])
pred_df.insert(0, "true_label", [class_mapping[i] for i in y_test])
pred_df.insert(1, "predicted_label", [class_mapping[i] for i in y_pred])
pred_df.to_csv(os.path.join(METRICS_DIR, "test_predictions.csv"), index=False)

print("=" * 70)
print("SAVED FILES")
print("=" * 70)
print(f"  {FIGURES_DIR}/confusion_matrix_counts.png, confusion_matrix_normalized.png, roc_curves_test.png")
print(f"  {METRICS_DIR}/test_metrics.json, classification_report_test.txt,")
print(f"  {METRICS_DIR}/per_class_metrics_test.csv, confusion_matrix_test.csv, test_predictions.csv")
print("\nEVALUATION COMPLETE. Next step: 06_explain_model.py")
print("Reminder: do not go back and re-tune the model based on these test results.")
print("=" * 70)