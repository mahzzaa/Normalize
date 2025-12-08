# compare_lstm_knn_xgb.py
#
# Compare:
#   (1) LSTM autoencoder (reconstruction error + threshold)
#   (2) KNN on raw flattened windows
#   (3) XGBoost on LSTM latent features

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model, Model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ------------------------------------------------------------
# 1. Load labeled dataset (normal + synthetic abnormal)
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
json_path = os.path.join(BASE_DIR, "sensor_data_14th_may_1600_windows_with_labels.json")

with open(json_path, "r") as f:
    payload = json.load(f)

X_all = np.array(payload["windows"], dtype="float32")   # shape: (N, seq_len, num_features)
y_all = np.array(payload["labels"], dtype="int32")      # 0 = normal, 1 = abnormal

num_samples, seq_len, num_features = X_all.shape

print("Loaded labeled dataset:")
print("  X_all shape:", X_all.shape)
print("  y_all shape:", y_all.shape)
print("  #normal (0):   ", int((y_all == 0).sum()))
print("  #abnormal (1):", int((y_all == 1).sum()))

# ------------------------------------------------------------
# 2. Load trained LSTM autoencoder and evaluate it
# ------------------------------------------------------------

model_path = os.path.join(BASE_DIR, "gait_lstm_autoencoder.h5")
autoencoder = load_model(model_path, compile=False)
print("\nLoaded trained autoencoder from:", model_path)

# Reconstruction error for all samples
X_recon = autoencoder.predict(X_all)
errors_all = np.mean((X_all - X_recon) ** 2, axis=(1, 2))

# Use only normal samples to set threshold
mask_normal = (y_all == 0)
normal_errors = errors_all[mask_normal]

threshold = np.percentile(normal_errors, 95)
print("\n[LSTM] Selected threshold (95th percentile of normal errors):", float(threshold))

# Predict: error > threshold => abnormal (1), else normal (0)
y_pred_lstm = (errors_all > threshold).astype(int)

acc_lstm = accuracy_score(y_all, y_pred_lstm)
cm_lstm = confusion_matrix(y_all, y_pred_lstm)
report_lstm = classification_report(y_all, y_pred_lstm, digits=3)

print("\n[LSTM] Classification accuracy:", float(acc_lstm))
print("[LSTM] Confusion matrix:\n", cm_lstm)
print("[LSTM] Classification report:\n", report_lstm)

# ------------------------------------------------------------
# 3. KNN on raw flattened windows (5-fold CV)
# ------------------------------------------------------------

X_flat = X_all.reshape(num_samples, seq_len * num_features)

scaler = StandardScaler()
X_flat_scaled = scaler.fit_transform(X_flat)

knn = KNeighborsClassifier(
    n_neighbors=5,
    metric="euclidean",
    weights="uniform"
)

print("\nEvaluating KNN on raw windows with 5-fold cross-validation...")
scores_knn = cross_val_score(knn, X_flat_scaled, y_all, cv=5, scoring="accuracy")

knn_mean_acc = scores_knn.mean()
knn_std_acc = scores_knn.std()

print("[KNN] Fold accuracies:", scores_knn)
print("[KNN] Mean accuracy:", float(knn_mean_acc))
print("[KNN] Std of accuracy:", float(knn_std_acc))

# Train final KNN on all data (optional, for deployment)
knn.fit(X_flat_scaled, y_all)
print("\n[KNN] Trained final KNN classifier on all data.")

# ------------------------------------------------------------
# 4. XGBoost on LSTM latent features (5-fold CV)
# ------------------------------------------------------------

# Build encoder model: output = encoder LSTM layer
encoder = Model(
    inputs=autoencoder.input,
    outputs=autoencoder.get_layer("encoder_lstm").output
)

# Latent features for all windows
Z_all = encoder.predict(X_all)   # shape: (N, latent_dim)
print("\nLatent features shape (Z_all):", Z_all.shape)

# Optional: you can also concatenate reconstruction error as an extra feature
# features_xgb = np.column_stack([Z_all, errors_all.reshape(-1, 1)])
features_xgb = Z_all

xgb_clf = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\nEvaluating XGBoost on LSTM latent features with 5-fold cross-validation...")
scores_xgb = cross_val_score(xgb_clf, features_xgb, y_all, cv=cv, scoring="accuracy")

xgb_mean_acc = scores_xgb.mean()
xgb_std_acc = scores_xgb.std()

print("[XGB] Fold accuracies:", scores_xgb)
print("[XGB] Mean accuracy:", float(xgb_mean_acc))
print("[XGB] Std of accuracy:", float(xgb_std_acc))

# Train final XGBoost on all data (optional)
xgb_clf.fit(features_xgb, y_all)
xgb_model_path = os.path.join(BASE_DIR, "xgb_on_lstm_latent.json")
xgb_clf.save_model(xgb_model_path)
print("\n[XGB] Trained final XGBoost classifier and saved to:", xgb_model_path)

# ------------------------------------------------------------
# 5. Visuals: confusion matrix (LSTM) + accuracy comparison
# ------------------------------------------------------------

# 5.1 LSTM confusion matrix heatmap
cm = cm_lstm
classes = ["Normal (0)", "Abnormal (1)"]

plt.figure()
plt.imshow(cm, interpolation="nearest")
plt.title("LSTM Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, str(cm[i, j]),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black"
        )

plt.ylabel("True label")
plt.xlabel("Predicted label")
cm_path = os.path.join(BASE_DIR, "lstm_confusion_matrix.png")
plt.tight_layout()
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✅ Saved LSTM confusion matrix figure to: {cm_path}")

# 5.2 Accuracy comparison bar chart (LSTM vs KNN vs XGB)
models = ["LSTM (AE+thr)", "KNN raw", "XGB latent"]
accuracies = [acc_lstm, knn_mean_acc, xgb_mean_acc]

plt.figure()
bars = plt.bar(models, accuracies)
plt.ylim(0.6, 1.0)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01,
             f"{acc:.3f}", ha="center", va="bottom")

acc_fig_path = os.path.join(BASE_DIR, "accuracy_comparison_lstm_knn_xgb.png")
plt.savefig(acc_fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved accuracy comparison figure to: {acc_fig_path}")
