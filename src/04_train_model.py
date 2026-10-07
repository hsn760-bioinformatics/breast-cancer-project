"""
04_train_model.py

Purpose:
- Load the ANOVA-selected train/val/test features
- Scale features using StandardScaler FIT ON TRAIN ONLY
- Build a Deep Neural Network for 6-class PAM50 subtype classification
- Handle class imbalance via class weights (not oversampling, since
  synthetic gene-expression profiles are hard to justify biologically)
- Train with early stopping + model checkpointing
- Save the trained model, the scaler, and training curves
"""

import os
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed; figures are saved to disk
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# ---------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ---------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
FIGURES_DIR = os.path.join("results", "figures")
METRICS_DIR = os.path.join("results", "metrics")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "breast_cancer_subtype_dnn.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")

# ---------------------------------------------------------------
# 2. Load ANOVA-selected features and targets
# ---------------------------------------------------------------
print("Loading selected features and targets...")
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_selected.csv"))
X_val = pd.read_csv(os.path.join(PROCESSED_DIR, "X_val_selected.csv"))
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_selected.csv"))

y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv"))["target"].values
y_val = pd.read_csv(os.path.join(PROCESSED_DIR, "y_val.csv"))["target"].values
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv"))["target"].values

n_features = X_train.shape[1]
n_classes = len(np.unique(y_train))
print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
print(f"Number of input features: {n_features}")
print(f"Number of classes: {n_classes}\n")

# ---------------------------------------------------------------
# 3. Feature scaling -- FIT ON TRAIN ONLY, applied to val/test
# ---------------------------------------------------------------
print("Scaling features (StandardScaler fit on training set only)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_val_scaled = scaler.transform(X_val.values)
X_test_scaled = scaler.transform(X_test.values)

joblib.dump(scaler, SCALER_PATH)
print(f"Scaler saved to: {SCALER_PATH}\n")

# ---------------------------------------------------------------
# 4. Class weights (counter class imbalance, e.g. 'Normal' has far
#    fewer samples than 'LumA')
# ---------------------------------------------------------------
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train,
)
class_weight_dict = {int(cls): float(w) for cls, w in zip(np.unique(y_train), class_weights_array)}
print("Class weights (balanced):")
for cls, w in class_weight_dict.items():
    print(f"  class {cls}: weight {w:.3f}")
print()

# ---------------------------------------------------------------
# 5. Build the DNN
#
#    Input -> Dense(128) -> BatchNorm -> ReLU -> Dropout(0.4)
#          -> Dense(64)  -> BatchNorm -> ReLU -> Dropout(0.3)
#          -> Dense(n_classes, softmax)
# ---------------------------------------------------------------
def build_model(input_dim: int, n_classes: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.4),

        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),

        layers.Dense(n_classes, activation="softmax"),
    ])
    return model

model = build_model(n_features, n_classes)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()
print()

# ---------------------------------------------------------------
# 6. Callbacks: early stopping + best-model checkpointing
# ---------------------------------------------------------------
early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=25,
    restore_best_weights=True,
    verbose=1,
)
checkpoint = callbacks.ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

# ---------------------------------------------------------------
# 7. Train
# ---------------------------------------------------------------
print("Starting training...\n")
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=200,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop, checkpoint],
    verbose=2,
)

# Save the final (best-weights-restored) model explicitly
model.save(MODEL_PATH)
print(f"\nFinal model saved to: {MODEL_PATH}")

# ---------------------------------------------------------------
# 8. Save training history
# ---------------------------------------------------------------
history_df = pd.DataFrame(history.history)
history_path = os.path.join(METRICS_DIR, "training_history.csv")
history_df.to_csv(history_path, index=False)
print(f"Training history saved to: {history_path}")

# ---------------------------------------------------------------
# 9. Plot and save training/validation curves
# ---------------------------------------------------------------
epochs_ran = len(history_df)
print(f"Training stopped after {epochs_ran} epochs "
      f"(early stopping patience=25 on val_loss).\n")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history_df["loss"], label="Train Loss")
axes[0].plot(history_df["val_loss"], label="Validation Loss")
axes[0].set_title("Loss over epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(history_df["accuracy"], label="Train Accuracy")
axes[1].plot(history_df["val_accuracy"], label="Validation Accuracy")
axes[1].set_title("Accuracy over epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()
curves_path = os.path.join(FIGURES_DIR, "training_validation_curves.png")
plt.savefig(curves_path, dpi=150)
plt.close()
print(f"Training/validation curves saved to: {curves_path}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE. Next step: 05_evaluate_model.py")
print("=" * 70)