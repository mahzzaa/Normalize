import os
import json
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model, Model
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

# Optional: only if you want to add SignalMask features
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

from xgboost import XGBClassifier


def main():
    base_dir = os.path.dirname(__file__)

    json_path = os.path.join(base_dir, "physionet_windows_with_labels.json")
    model_path = os.path.join(base_dir, "physionet_lstm_autoencoder.h5")

    with open(json_path, "r") as f:
        payload = json.load(f)

    X_all = np.array(payload["windows"], dtype="float32")
    y_all = np.array(payload["labels"], dtype="int32")

    print("Loaded labeled dataset:")
    print("  X_all:", X_all.shape, " y_all:", y_all.shape)

    # Load AE (no compile to avoid mse deserialization problems)
    autoencoder = load_model(model_path, compile=False)
    print("\nLoaded autoencoder:", model_path)

    # Build encoder model to extract latent vector Z
    encoder = Model(
        inputs=autoencoder.input,
        outputs=autoencoder.get_layer("encoder_lstm").output
    )

    Z_all = encoder.predict(X_all, verbose=0)  # (N, latent_dim)
    X_recon = autoencoder.predict(X_all, verbose=0)
    errors_all = np.mean((X_all - X_recon) ** 2, axis=(1, 2))  # (N,)

    # Features = [Z , reconstruction_error]
    feats = np.concatenate([Z_all, errors_all.reshape(-1, 1)], axis=1)
    print("\nBase features shape (Z + error):", feats.shape)

    # Optionally merge SignalMask-style CSV features if available
    sm_path = os.path.join(base_dir, "signalmask_window_features.csv")
    if HAS_PANDAS and os.path.exists(sm_path):
        df = pd.read_csv(sm_path)
        if "window_id" in df.columns:
            df = df.drop(columns=["window_id"])
        sm_feats = df.values.astype(np.float32)

        if sm_feats.shape[0] == feats.shape[0]:
            feats = np.concatenate([feats, sm_feats], axis=1)
            print("✅ Added SignalMask features:", sm_feats.shape, "=> total:", feats.shape)
        else:
            print("⚠️ SignalMask rows do not match windows. Skipping SignalMask merge.")
    else:
        print("No SignalMask CSV merged (missing file or pandas).")

    # ----- LSTM AE+Threshold baseline -----
    errors_normal = errors_all[y_all == 0]
    threshold = np.percentile(errors_normal, 95)
    y_pred_lstm = (errors_all > threshold).astype(int)
    lstm_acc = accuracy_score(y_all, y_pred_lstm)
    lstm_f1 = f1_score(y_all, y_pred_lstm)
    print(f"\n[LSTM AE+thr] Acc={lstm_acc:.4f}  F1={lstm_f1:.4f}  thr={threshold:.6f}")

    # ----- XGBoost on (Z + error [+ signalmask]) with CV -----
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb_accs = cross_val_score(xgb, feats, y_all, cv=skf, scoring="accuracy")
    xgb_f1s = cross_val_score(xgb, feats, y_all, cv=skf, scoring="f1")

    xgb_acc_mean = float(xgb_accs.mean())
    xgb_acc_std = float(xgb_accs.std())
    xgb_f1_mean = float(xgb_f1s.mean())
    xgb_f1_std = float(xgb_f1s.std())

    print("\n[XGB on LSTM features] 5-fold CV")
    print("  Acc fold:", xgb_accs)
    print(f"  Acc mean={xgb_acc_mean:.4f}  std={xgb_acc_std:.4f}")
    print(f"  F1  mean={xgb_f1_mean:.4f}  std={xgb_f1_std:.4f}")

    # ----- Plot accuracy comparison -----
    labels = ["LSTM (AE+thr)", "XGB (Z+err CV mean)"]
    values = [lstm_acc, xgb_acc_mean]

    plt.figure()
    bars = plt.bar(labels, values)
    plt.ylim(0, 1.0)
    plt.title("PhysioNet Accuracy: LSTM vs LSTM→XGBoost")
    plt.ylabel("Accuracy")

    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}", ha="center")

    out_path = os.path.join(base_dir, "physionet_accuracy_lstm_xgb.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("\n✅ Saved plot to:", out_path)


if __name__ == "__main__":
    main()
