# train_autoencoder.py

import os
import json
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam

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

# ---------- 4. Save the trained model ----------

model_path = os.path.join(BASE_DIR, "gait_lstm_autoencoder.h5")
autoencoder.save(model_path)
print(f"\n✅ Saved trained model to: {model_path}")

# ---------- 5. Optional: compute reconstruction error on training data ----------

# This is just to get a feeling of error range on "normal" gait
reconstructed = autoencoder.predict(X)
# Mean squared error per window
errors = np.mean(np.square(X - reconstructed), axis=(1, 2))

print("\nReconstruction error stats on training data:")
print("  min :", float(errors.min()))
print("  max :", float(errors.max()))
print("  mean:", float(errors.mean()))
print("  std :", float(errors.std()))
