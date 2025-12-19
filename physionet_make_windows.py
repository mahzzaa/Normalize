# physionet_make_windows.py
import os
import glob
import json
import numpy as np

BASE_DIR = os.path.dirname(__file__)

# Change this to the folder where you extracted PhysioNet files
DATA_DIR = os.path.join(BASE_DIR, "gait-data")  # e.g. Normalize/gait-data

WIN_LEN = 100
HOP = 100  # no overlap

X_windows = []
y_labels = []

for path in glob.glob(os.path.join(DATA_DIR, "*.txt")):
    name = os.path.basename(path).lower()

    # Label rule: pd* = abnormal (1), others (y*, o*) = normal (0)
    label = 1 if name.startswith("pd") else 0

    data = np.loadtxt(path)          # 2 columns: time, stride_interval
    stride = data[:, 1].astype(np.float32)

    for start in range(0, len(stride) - WIN_LEN + 1, HOP):
        w = stride[start:start + WIN_LEN]
        X_windows.append(w.reshape(WIN_LEN, 1))
        y_labels.append(label)

X = np.stack(X_windows)              # (N, 100, 1)
y = np.array(y_labels, dtype=np.int32)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("abnormal (1):", int((y == 1).sum()))
print("normal (0):", int((y == 0).sum()))

# Save to JSON (same style as your project)
out_path = os.path.join(BASE_DIR, "physionet_windows_with_labels.json")
payload = {"windows": X.tolist(), "labels": y.tolist()}
with open(out_path, "w") as f:
    json.dump(payload, f)

print("Saved:", out_path)
