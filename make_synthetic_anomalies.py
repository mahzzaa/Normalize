# make_synthetic_anomalies.py

import os
import json
import numpy as np

# ---------- Load original (normal) windowed data ----------

BASE_DIR = os.path.dirname(__file__)
in_path = os.path.join(BASE_DIR, "sensor_data_14th_may_1600_windows.json")

with open(in_path, "r") as f:
    payload = json.load(f)

X = np.array(payload["windows"], dtype="float32")   # shape: (N, seq_len, num_features)
num_windows, seq_len, num_features = X.shape

print("Loaded windows:", X.shape)

# For reproducibility
np.random.seed(42)

# ---------- Simple helpers to create synthetic anomalies ----------

def add_spikes(window, num_spikes=5, spike_scale=2.0):
    """
    Add a few large spikes to random time steps and features.
    This simulates sudden abnormal sensor readings.
    """
    w = window.copy()
    T, F = w.shape

    for _ in range(num_spikes):
        t = np.random.randint(0, T)
        f = np.random.randint(0, F)
        # Add a positive or negative spike
        w[t, f] += spike_scale * (np.random.rand() - 0.5)
    return w


def shuffle_segment(window, seg_len=10):
    """
    Randomly shuffle a short temporal segment.
    This breaks the natural temporal order of gait.
    """
    w = window.copy()
    T, F = w.shape
    if T <= seg_len:
        return w

    start = np.random.randint(0, T - seg_len)
    end = start + seg_len
    segment = w[start:end].copy()
    np.random.shuffle(segment)   # shuffle along time dimension
    w[start:end] = segment
    return w


def scale_feature(window, scale_min=1.5, scale_max=3.0):
    """
    Strongly scale one feature (e.g., one sensor channel).
    This simulates abnormally high or low readings for that sensor.
    """
    w = window.copy()
    T, F = w.shape
    f = np.random.randint(0, F)
    scale = np.random.uniform(scale_min, scale_max)
    w[:, f] *= scale
    return w


def make_anomaly(window):
    """
    Combine several perturbations to create an abnormal window.
    We randomly decide which types of corruption to apply.
    """
    w = window.copy()
    if np.random.rand() < 0.7:
        w = add_spikes(w, num_spikes=5)
    if np.random.rand() < 0.7:
        w = shuffle_segment(w, seg_len=10)
    if np.random.rand() < 0.5:
        w = scale_feature(w)
    return w


# ---------- Build new dataset with normal + synthetic abnormal ----------

windows_out = []
labels_out = []

# All original windows are treated as normal (label = 0)
for i in range(num_windows):
    w = X[i]
    windows_out.append(w.tolist())
    labels_out.append(0)

    # With some probability, create one synthetic abnormal version
    if np.random.rand() < 0.7:
        w_abn = make_anomaly(w)
        windows_out.append(w_abn.tolist())
        labels_out.append(1)

windows_out = np.array(windows_out, dtype="float32")
labels_out = np.array(labels_out, dtype="int32")

print("Original normal windows:", num_windows)
print("Total windows (normal + synthetic abnormal):", windows_out.shape[0])
print("Labels 0 count (normal):", int((labels_out == 0).sum()))
print("Labels 1 count (abnormal):", int((labels_out == 1).sum()))

# ---------- Save to a new JSON file ----------

out_path = os.path.join(BASE_DIR, "sensor_data_14th_may_1600_windows_with_labels.json")

payload_out = {
    "windows": windows_out.tolist(),
    "labels": labels_out.tolist(),
}

with open(out_path, "w") as f:
    json.dump(payload_out, f)

print("\n✅ Saved new dataset with synthetic anomalies to:")
print("   ", out_path)
