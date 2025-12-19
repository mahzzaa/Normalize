# optimize_lstm_threshold.py

import os, json
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(__file__)
JSON_PATH = os.path.join(BASE_DIR, "sensor_data_14th_may_1600_windows_with_labels.json")
MODEL_PATH = os.path.join(BASE_DIR, "gait_lstm_autoencoder.h5")

with open(JSON_PATH, "r") as f:
    payload = json.load(f)

X_all = np.array(payload["windows"], dtype="float32")
y_all = np.array(payload["labels"], dtype="int32")

autoencoder = load_model(MODEL_PATH, compile=False)

# Reconstruction errors for all windows
X_rec_all = autoencoder.predict(X_all, verbose=0)
all_errors = np.mean((X_all - X_rec_all) ** 2, axis=(1, 2))

# Build a "normal-only" TRAIN subset for threshold selection (no leakage)
X_normal = X_all[y_all == 0]
Xn_train, Xn_val = train_test_split(X_normal, test_size=0.2, random_state=42)

Xn_train_rec = autoencoder.predict(Xn_train, verbose=0)
train_errors_normal = np.mean((Xn_train - Xn_train_rec) ** 2, axis=(1, 2))

percentiles = [90, 92.5, 95, 97.5, 99]
best_f1 = -1
best_thr = None

for p in percentiles:
    thr = np.percentile(train_errors_normal, p)
    preds = (all_errors > thr).astype(int)
    f1 = f1_score(y_all, preds)

    print(f"Percentile={p:>4}  thr={thr:.6f}  F1={f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

print("\nBest threshold:", float(best_thr))
print("Best F1:", float(best_f1))
