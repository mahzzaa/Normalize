# train_autoencoder.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------- 1. Load windowed data from JSON ----------

BASE_DIR = os.path.dirname(__file__)
json_path = os.path.join(BASE_DIR, "sensor_data_14th_may_1600_windows.json")

with open(json_path, "r") as f:
    payload = json.load(f)

# X shape: (num_windows, seq_len, 3)
X = np.array(payload["windows"], dtype="float32")

print("Loaded windows:")
print("  X shape:", X.shape)  # e.g. (420, 100, 3)

num_samples, seq_len, num_features = X.shape

# Optional: labels for classification (0 = normal, 1 = abnormal)
y = None
if "labels" in payload:
    y = np.array(payload["labels"], dtype="int32")
    if y.shape[0] != num_samples:
        raise ValueError(
            f"Number of labels ({y.shape[0]}) does not match number of windows ({num_samples})."
        )
    print("  Loaded labels with shape:", y.shape)
else:
    print("  No 'labels' field found in JSON. Classification accuracy will be skipped.")

# ---------- 2. Build LSTM Autoencoder model ----------

# Latent dimension (size of the encoded vector)
latent_dim = 64

# Input shape: (time_steps, features) = (seq_len, 3)
inputs = Input(shape=(seq_len, num_features), name="input_sequence")

# Encoder: compress sequence into a single vector
encoded = LSTM(latent_dim, name="encoder_lstm")(inputs)

# Repeat the encoded vector seq_len times to prepare for decoding
bottleneck = RepeatVector(seq_len, name="repeat_vector")(encoded)

# Decoder: reconstruct the original sequence
decoded = LSTM(latent_dim, return_sequences=True, name="decoder_lstm")(bottleneck)

# TimeDistributed(Dense) outputs a 3D tensor with shape (batch, time, features)
outputs = TimeDistributed(Dense(num_features), name="decoder_output")(decoded)

# Full autoencoder model: input -> reconstructed sequence
autoencoder = Model(inputs, outputs, name="gait_lstm_autoencoder")

autoencoder.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="mse"   # mean squared error between input and reconstructed output
)

print("\nModel summary:")
autoencoder.summary()

# ---------- 3. Train the autoencoder ----------

EPOCHS = 40
BATCH_SIZE = 32
VAL_SPLIT = 0.2

print("\nStarting training...")
history = autoencoder.fit(
    X, X,                     # input = target (autoencoder learns to reconstruct X)
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT,
    shuffle=True
)

print("\nTraining finished.")

# ---------- 3.1 Plot training / validation loss ----------

plt.figure()
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("LSTM Autoencoder Reconstruction Loss")
plt.legend()
plot_path = os.path.join(BASE_DIR, "loss_curve.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved loss curve to: {plot_path}")
plt.close()

# ---------- 4. Save the trained model ----------

model_path = os.path.join(BASE_DIR, "gait_lstm_autoencoder.h5")
autoencoder.save(model_path)
print(f"\n✅ Saved trained model to: {model_path}")

# ---------- 5. Reconstruction error on training data ----------

reconstructed = autoencoder.predict(X)
# Mean squared error per window
errors = np.mean(np.square(X - reconstructed), axis=(1, 2))

print("\nReconstruction error stats on training data:")
print("  min :", float(errors.min()))
print("  max :", float(errors.max()))
print("  mean:", float(errors.mean()))
print("  std :", float(errors.std()))

# ---------- 6. Optional: classification using reconstruction error ----------

if y is not None:
    # Assume: y = 0 (normal), y = 1 (abnormal)

    # Select only normal windows to set threshold
    mask_normal = (y == 0)
    if not np.any(mask_normal):
        print("\n[Classification] No normal samples (label 0) found. "
              "Cannot compute threshold-based accuracy.")
    else:
        normal_errors = errors[mask_normal]

        # e.g., threshold = 95th percentile of normal reconstruct‍ion error
        threshold = np.percentile(normal_errors, 95)
        print("\n[Classification] Selected threshold (95th percentile of normal errors):",
              float(threshold))

        # Predict: error > threshold → abnormal (1), else normal (0)
        y_pred = (errors > threshold).astype(int)

        acc = accuracy_score(y, y_pred)
        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred, digits=3)

        print("\n[Classification] Accuracy based on reconstruction error:")
        print("  Accuracy:", float(acc))
        print("Confusion matrix:\n", cm)
        print("Classification report:\n", report)
else:
    print("\nNo labels provided → only reconstruction loss is reported, "
          "classification accuracy is not computed.")
