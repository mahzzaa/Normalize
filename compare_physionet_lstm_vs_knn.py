import os
import json
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score


def main():
    base_dir = os.path.dirname(__file__)

    json_path = os.path.join(base_dir, "physionet_windows_with_labels.json")
    model_path = os.path.join(base_dir, "physionet_lstm_autoencoder.h5")

    with open(json_path, "r") as f:
        payload = json.load(f)

    X_all = np.array(payload["windows"], dtype="float32")  # (N,100,1)
    y_all = np.array(payload["labels"], dtype="int32")     # (N,)

    print("Loaded labeled dataset:")
    print("  X_all:", X_all.shape, " y_all:", y_all.shape)

    # ----- LSTM (AE + threshold) classification -----
    autoencoder = load_model(model_path, compile=False)
    print("\nLoaded autoencoder:", model_path)

    X_recon = autoencoder.predict(X_all, verbose=0)
    errors_all = np.mean((X_all - X_recon) ** 2, axis=(1, 2))

    errors_normal = errors_all[y_all == 0]
    threshold = np.percentile(errors_normal, 95)  # change to best threshold if you found one
    y_pred = (errors_all > threshold).astype(int)

    lstm_acc = accuracy_score(y_all, y_pred)
    lstm_f1 = f1_score(y_all, y_pred)

    print(f"\n[LSTM AE+thr] threshold(95%)={threshold:.6f}")
    print(f"[LSTM AE+thr] Accuracy={lstm_acc:.4f}  F1={lstm_f1:.4f}")

    # ----- KNN baseline on raw windows -----
    # Flatten each window into 1D vector: (N,100,1) -> (N,100)
    X_flat = X_all.reshape(X_all.shape[0], -1)

    knn = KNeighborsClassifier(n_neighbors=5)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    knn_accs = cross_val_score(knn, X_flat, y_all, cv=skf, scoring="accuracy")
    knn_f1s = cross_val_score(knn, X_flat, y_all, cv=skf, scoring="f1")

    knn_acc_mean = float(knn_accs.mean())
    knn_acc_std = float(knn_accs.std())
    knn_f1_mean = float(knn_f1s.mean())
    knn_f1_std = float(knn_f1s.std())

    print("\n[KNN raw] 5-fold CV")
    print("  Acc fold:", knn_accs)
    print(f"  Acc mean={knn_acc_mean:.4f}  std={knn_acc_std:.4f}")
    print(f"  F1  mean={knn_f1_mean:.4f}  std={knn_f1_std:.4f}")

    # ----- Plot accuracy comparison -----
    labels = ["LSTM (AE+thr)", "KNN raw (CV mean)"]
    values = [lstm_acc, knn_acc_mean]

    plt.figure()
    bars = plt.bar(labels, values)
    plt.ylim(0, 1.0)
    plt.title("PhysioNet Model Accuracy Comparison")
    plt.ylabel("Accuracy")

    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")

    out_path = os.path.join(base_dir, "physionet_accuracy_lstm_knn.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("\n✅ Saved plot to:", out_path)


if __name__ == "__main__":
    main()
