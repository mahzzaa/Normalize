# compare_lstm_vs_knn.py
#
# Compare classification accuracy of:
#   (1) LSTM autoencoder (reconstruction-error based)
#   (2) KNN classifier trained directly on window data

import os
import json
import numpy as np
import matplotlib.pyplot as plt   # <-- این خط مهم

from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


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
# 2. Evaluate LSTM autoencoder (already trained)
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
# 3. Evaluate KNN classifier on the same dataset
# ------------------------------------------------------------

# Flatten each window: (seq_len, num_features) -> (seq_len * num_features,)
X_flat = X_all.reshape(num_samples, seq_len * num_features)

# Standardize features (important for KNN distance)
scaler = StandardScaler()
X_flat_scaled = scaler.fit_transform(X_flat)

# Define a simple KNN classifier
knn = KNeighborsClassifier(
    n_neighbors=5,
    metric="euclidean",
    weights="uniform"
)

# Use 5-fold cross-validation for a fair estimate
print("\nEvaluating KNN with 5-fold cross-validation...")
scores = cross_val_score(knn, X_flat_scaled, y_all, cv=5, scoring="accuracy")

print("[KNN] Fold accuracies:", scores)
print("[KNN] Mean accuracy:", float(scores.mean()))
print("[KNN] Std of accuracy:", float(scores.std()))

# Optionally train KNN on all data (for later deployment / prediction)
knn.fit(X_flat_scaled, y_all)
print("\n[KNN] Trained final KNN classifier on all data.")


# ------------------------------------------------------------
# 4. Simple bar chart: LSTM vs KNN accuracy
# ------------------------------------------------------------

models = ["LSTM (AE)", "KNN"]
accuracies = [acc_lstm, scores.mean()]

plt.figure()
plt.bar(models, accuracies)
plt.ylim(0.6, 1.0)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison: LSTM Autoencoder vs KNN")

for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f"{acc:.3f}", ha="center")

bar_path = os.path.join(BASE_DIR, "accuracy_comparison_lstm_knn.png")
plt.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✅ Saved accuracy comparison figure to: {bar_path}")

# ------------------------------------------------------------
# 5. Confusion matrix heatmap for LSTM
# ------------------------------------------------------------

cm = cm_lstm  # already computed above
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
print(f"✅ Saved LSTM confusion matrix figure to: {cm_path}")
