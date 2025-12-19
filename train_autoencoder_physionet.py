# train_autoencoder_physionet.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# -----------------------------
# 1) Load PhysioNet windows JSON
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
json_path = os.path.join(BASE_DIR, "physionet_windows_with_labels.json")

with open(json_path, "r") as f:
    payload = json.load(f)

X_all = np.array(payload["windows"], dtype="float32")  # (N, 100, 1)
y_all = np.array(payload["labels"], dtype="int32")     # (N,)

print("Loaded labeled dataset:")
print("  X_all shape:", X_all.shape)
print("  y_all shape:", y_all.shape)
print("  #normal (0):   ", int((y_all == 0).sum()))
print("  #abnormal (1): ", int((y_all == 1).sum()))

num_samples, seq_len, num_features = X_all.shape

# -----------------------------
# 2) Train AE only on NORMAL windows
# -----------------------------
X_train = X_all[y_all == 0]
print("\nTraining on normal only:")
print("  X_train shape:", X_train.shape)

# -----------------------------
# 3) Build LSTM Autoencoder
# -----------------------------
latent_dim = 64

inputs = Input(shape=(seq_len, num_features), name="input_sequence")
encoded = LSTM(latent_dim, name="encoder_lstm")(inputs)
bottleneck = RepeatVector(seq_len, name="repeat_vector")(encoded)
decoded = LSTM(latent_dim, return_sequences=True, name="decoder_lstm")(bottleneck)
outputs = TimeDistributed(Dense(num_features), name="decoder_output")(decoded)

autoencoder = Model(inputs, outputs, name="physionet_lstm_autoencoder")
autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")

print("\nModel summary:")
autoencoder.summary()

# -----------------------------
# 4) Train
# -----------------------------
EPOCHS = 40
BATCH_SIZE = 32
VAL_SPLIT = 0.2

print("\nStarting training...")
history = autoencoder.fit(
    X_train, X_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT,
    shuffle=True
)
print("\nTraining finished.")

# -----------------------------
# 5) Save model + loss curve
# -----------------------------
model_path = os.path.join(BASE_DIR, "physionet_lstm_autoencoder.h5")
autoencoder.save(model_path)
print(f"\n✅ Saved trained model to: {model_path}")

plt.figure()
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("PhysioNet LSTM Autoencoder Reconstruction Loss")
plt.legend()

loss_plot_path = os.path.join(BASE_DIR, "physionet_loss_curve.png")
plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved loss curve to: {loss_plot_path}")

# -----------------------------
# 6) Compute reconstruction error for ALL windows
# -----------------------------
X_recon = autoencoder.predict(X_all)
errors_all = np.mean((X_all - X_recon) ** 2, axis=(1, 2))

print("\nReconstruction error stats (ALL labeled data):")
print("  min :", float(errors_all.min()))
print("  max :", float(errors_all.max()))
print("  mean:", float(errors_all.mean()))
print("  std :", float(errors_all.std()))

# -----------------------------
# 7) Choose threshold from NORMAL errors only
# -----------------------------
errors_normal = errors_all[y_all == 0]
threshold = np.percentile(errors_normal, 95)  # start with 95%
print("\nSelected threshold = 95th percentile of NORMAL errors:", float(threshold))

# Predict: error > thr => abnormal (1)
y_pred = (errors_all > threshold).astype(int)

acc = accuracy_score(y_all, y_pred)
cm = confusion_matrix(y_all, y_pred)

print("\n[Classification] Accuracy:", float(acc))
print("[Classification] Confusion matrix:\n", cm)
print("[Classification] Report:\n", classification_report(y_all, y_pred, digits=3))

# -----------------------------
# 8) Save confusion matrix image
# -----------------------------
plt.figure()
plt.imshow(cm)
plt.title("PhysioNet Confusion Matrix (AE+Threshold)")
plt.xlabel("Predicted")
plt.ylabel("True")

for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")

cm_path = os.path.join(BASE_DIR, "physionet_confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved confusion matrix to: {cm_path}")
