import os
import json
import numpy as np
from sklearn.metrics import f1_score
from tensorflow.keras.models import load_model

def main():
    base_dir = os.path.dirname(__file__)

    # Change these names if your files are named differently
    json_path = os.path.join(base_dir, "physionet_windows_with_labels.json")
    model_path = os.path.join(base_dir, "physionet_lstm_autoencoder.h5")

    # Load labeled windows
    with open(json_path, "r") as f:
        payload = json.load(f)

    X_all = np.array(payload["windows"], dtype="float32")  # (N, 100, 1)
    y_all = np.array(payload["labels"], dtype="int32")     # (N,)

    print("Loaded labeled dataset:")
    print("  X_all shape:", X_all.shape)
    print("  y_all shape:", y_all.shape)
    print("  #normal (0):", int((y_all == 0).sum()))
    print("  #abnormal(1):", int((y_all == 1).sum()))

    # Load autoencoder without compiling (avoids mse load issues)
    autoencoder = load_model(model_path, compile=False)
    print("\nLoaded model:", model_path)

    # Reconstruction errors
    X_recon = autoencoder.predict(X_all, verbose=0)
    errors_all = np.mean((X_all - X_recon) ** 2, axis=(1, 2))

    # Only normal errors for threshold selection
    errors_normal = errors_all[y_all == 0]

    percentiles = [90, 92.5, 95, 97.5, 99]
    best_f1 = -1
    best_thr = None
    best_p = None

    print("\nSearching best percentile threshold using F1...")
    for p in percentiles:
        thr = np.percentile(errors_normal, p)
        preds = (errors_all > thr).astype(int)
        f1 = f1_score(y_all, preds)

        print(f"Percentile={p:>5}  thr={thr:.6f}  F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_p = p

    print("\n✅ Best threshold:")
    print("  Best percentile:", best_p)
    print("  Best threshold :", best_thr)
    print("  Best F1        :", best_f1)

if __name__ == "__main__":
    main()
