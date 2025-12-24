import json
import argparse
import numpy as np
import pandas as pd


# --------------------------
# Helpers
# --------------------------
def safe_diff(x):
    if len(x) < 2:
        return np.array([0.0])
    return np.diff(x)

def rms(x):
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x)) + 1e-12)

def normalize01(x):
    x = np.asarray(x, dtype=np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def estimate_pronation_angle_deg(window_xyz):
    """
    If window has 3 axes: (T,3) -> estimate a tilt angle proxy (degrees).
    This is NOT medical pronation; it's a signal-based orientation proxy.
    """
    w = np.asarray(window_xyz, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] < 3:
        return np.nan

    x = w[:, 0]
    y = w[:, 1]
    z = w[:, 2]

    # A common IMU-style tilt proxy (roll-like)
    # roll = atan2(y, z)
    ang = np.degrees(np.arctan2(y, z + 1e-12))
    return float(np.mean(ang))


def compute_metrics_from_window(window):
    """
    window shape:
      - (T,)  -> 1D signal (e.g., PhysioNet stride interval)
      - (T,1) -> same as 1D
      - (T,3) -> XYZ accel-like
    Returns 4 metrics:
      Step length (proxy), Symmetry (proxy), Loading rate (proxy), Pronation angle (proxy)
    """

    w = np.asarray(window, dtype=np.float32)

    # Make 1D version for some metrics
    if w.ndim == 2 and w.shape[1] == 1:
        s = w[:, 0]
    elif w.ndim == 1:
        s = w
    elif w.ndim == 2 and w.shape[1] >= 3:
        # use magnitude for 1D proxy signal
        s = np.linalg.norm(w[:, :3], axis=1)
    else:
        s = w.reshape(-1)

    s = np.asarray(s, dtype=np.float32)
    ds = safe_diff(s)

    # 1) Step length (proxy)
    # If you don’t have distance, a reasonable proxy is signal “energy” / amplitude.
    step_length_proxy = rms(s)

    # 2) Symmetry (proxy)
    # Compare first half vs second half similarity (correlation).
    half = len(s) // 2
    if half < 5:
        symmetry_proxy = np.nan
    else:
        a = s[:half]
        b = s[-half:]
        a_n = a - np.mean(a)
        b_n = b - np.mean(b)
        denom = (np.linalg.norm(a_n) * np.linalg.norm(b_n) + 1e-12)
        symmetry_proxy = float(np.dot(a_n, b_n) / denom)  # -1..1 (1 = very symmetric)

    # 3) Loading rate (proxy)
    # High positive slope (impact-like change). Use max positive derivative.
    loading_rate_proxy = float(np.max(ds)) if len(ds) else 0.0

    # 4) Pronation angle (proxy)
    # If you have XYZ, compute roll-like angle. If 1D, leave NaN.
    pronation_angle_proxy = estimate_pronation_angle_deg(w) if (w.ndim == 2 and w.shape[1] >= 3) else np.nan

    return step_length_proxy, symmetry_proxy, loading_rate_proxy, pronation_angle_proxy


def load_windows_from_json(json_path):
    """
    Supports your PhysioNet-style file:
      physionet_windows_with_labels.json

    Expected structure (one of these):
      A) {"windows": [...], "labels": [...]}
      B) [{"window": [...], "label": 0/1}, ...]
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "windows" in data:
        windows = data["windows"]
        labels = data.get("labels", [None] * len(windows))
        return windows, labels

    if isinstance(data, list):
        windows = []
        labels = []
        for item in data:
            if isinstance(item, dict) and "window" in item:
                windows.append(item["window"])
                labels.append(item.get("label", None))
            else:
                # if it's just raw windows list
                windows.append(item)
                labels.append(None)
        return windows, labels

    raise ValueError("Unsupported JSON structure for windows.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to windows JSON (e.g., physionet_windows_with_labels.json)")
    ap.add_argument("--out", default="metrics_table.csv", help="Output CSV path")
    ap.add_argument("--name", default="dataset", help="Dataset name to store in table")
    args = ap.parse_args()

    windows, labels = load_windows_from_json(args.input)

    rows = []
    for i, w in enumerate(windows):
        step_len, sym, load_rate, pron_angle = compute_metrics_from_window(w)
        rows.append({
            "dataset": args.name,
            "window_id": i + 1,
            "label": labels[i] if i < len(labels) else None,
            "step_length": step_len,
            "symmetry": sym,
            "loading_rate": load_rate,
            "pronation_angle": pron_angle
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print(f"✅ Saved metrics table to: {args.out}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
