# train_autoencoder.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------------------------------------------------
# 1. Load NORMAL windows (for training the autoencoder)
#    -> from: sensor_data_14th_may_1600_windows.json
# ------------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)

normal_json_path = os.path.join(BASE_DIR, "sensor_data_14th_may_1600_windows.json")
with open(normal_json_path, "r") as f:
    normal_payload = json.load(f)

X_normal = np.array(normal_payload["windows"], dtype="float32")   # shape: (N_normal, seq_len, features)

print("Loaded NORMAL windows (training data):")
print("  X_normal shape:", X_normal.shape)

num_normal, seq_len, num_features = X_normal.shape

# ------------------------------------------------------------------
# 2. Load dataset with labels (NORMAL + synthetic ABNORMAL)
#    -> from: sensor_data_14th_may_1600_windows_with_labels.json
# ------------------------------------------------------------------

labeled_json_path = os.path.join(BASE_DIR, "sensor_data_14th_may_1600_windows_with_labels.json")
with open(labeled_json_path, "r") as f:
    labeled_payload = json.load(f)

X_all = np.array(labeled_payload["windows"], dtype="float32")
y_all = np.array(labeled_payload["labels"], dtype="int32")   # 0 = normal, 1 = abnormal

print("\nLoaded LABELED dataset (for evaluation):")
print("  X_all shape:", X_all.shape)
print("  y_all shape:", y_all.shape)
print("  #normal (label=0):", int((y_all == 0).sum()))
print("  #abnormal (label=1):", int((y_all == 1).sum()))

# Basic consistency checks
if X_all.shape[1] != seq_len or X_all.shape[2] != num_features:
    raise ValueError("Shape mismatch between normal windows and labeled dataset.")
if X_all.shape[0] != y_all.shape[0]:
    raise ValueError("Number of windows and labels do not match in labeled dataset.")

# ------------------------------------------------------------------
# 3. Build LSTM Autoencoder model
# ------------------------------------------------------------------

latent_dim = 64  # size of the encoded vector

inputs = Input(shape=(seq_len, num_features), name="input_sequence")

# Encoder
encoded = LSTM(latent_dim, name="encoder_lstm")(inputs)

# Bottleneck
bottleneck = RepeatVector(seq_len, name="repeat_vector")(encoded)

# Decoder
decoded = LSTM(latent_dim, return_sequences=True, name="decoder_lstm")(bottleneck)
outputs = TimeDistributed(Dense(num_features), name="decoder_output")(decoded)

autoencoder = Model(inputs, outputs, name="gait_lstm_autoencoder")

autoencoder.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="mse"
)

print("\nModel summary:")
autoencoder.summary()

# ------------------------------------------------------------------
# 4. Train the autoencoder ONLY on NORMAL data
# ------------------------------------------------------------------

EPOCHS = 40
BATCH_SIZE = 32
VAL_SPLIT = 0.2

print("\nStarting training on NORMAL windows...")
history = autoencoder.fit(
    X_normal, X_normal,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT,
    shuffle=True
)

print("\nTraining finished.")

# ------------------------------------------------------------------
# 4.1 Plot training / validation loss
# ------------------------------------------------------------------

plt.figure()
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("LSTM Autoencoder Reconstruction Loss (Normal training data)")
plt.legend()
plot_path = os.path.join(BASE_DIR, "loss_curve.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved loss curve to: {plot_path}")
plt.close()

# Also print final loss values
final_train_loss = history.history["loss"][-1]
final_val_loss = history.history["val_loss"][-1]
print("\nFinal training loss (MSE):", float(final_train_loss))
print("Final validation loss (MSE):", float(final_val_loss))

# ------------------------------------------------------------------
# 5. Save the trained model
# ------------------------------------------------------------------

model_path = os.path.join(BASE_DIR, "gait_lstm_autoencoder.h5")
autoencoder.save(model_path)
print(f"\n✅ Saved trained model to: {model_path}")

# ------------------------------------------------------------------
# 6. Reconstruction error on LABELED dataset (normal + abnormal)
# ------------------------------------------------------------------

reconstructed_all = autoencoder.predict(X_all)
errors_all = np.mean(np.square(X_all - reconstructed_all), axis=(1, 2))

print("\nReconstruction error stats on LABELED data:")
print("  min :", float(errors_all.min()))
print("  max :", float(errors_all.max()))
print("  mean:", float(errors_all.mean()))
print("  std :", float(errors_all.std()))

# ------------------------------------------------------------------
# 7. Classification using reconstruction error
# ------------------------------------------------------------------

# Use only normal samples (label=0) in the LABELED dataset to set the threshold
mask_normal = (y_all == 0)
if not np.any(mask_normal):
    print("\n[Classification] No normal samples (label 0) found in labeled dataset.")
else:
    normal_errors = errors_all[mask_normal]

    # Threshold: 95th percentile of normal reconstruction error
    threshold = np.percentile(normal_errors, 95)
    print("\n[Classification] Selected threshold (95th percentile of normal errors):",
          float(threshold))

    # Predict: error > threshold → abnormal (1), else normal (0)
    y_pred = (errors_all > threshold).astype(int)

    acc = accuracy_score(y_all, y_pred)
    cm = confusion_matrix(y_all, y_pred)
    report = classification_report(y_all, y_pred, digits=3)

    print("\n[Classification] Accuracy based on reconstruction error:")
    print("  Accuracy:", float(acc))
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", report)
