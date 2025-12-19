import os
import numpy as np
import pandas as pd

# -----------------------------
# Settings (match your project)
# -----------------------------
BASE_DIR = os.path.dirname(__file__)

CSV_PATH = os.path.join(BASE_DIR, "sensor_data_14th_may_1600_normalized.csv")
OUT_PATH = os.path.join(BASE_DIR, "signalmask_window_features.csv")

WIN_LEN = 100   # same as your window length
HOP = 100       # same hop (100 = no overlap, 50 = 50% overlap)

# Thresholds (tune if needed)
FLAT_STD_EPS = 1e-4        # "flat" if rolling std below this
FLAT_WIN = 10              # rolling window size for flat detection
SAT_THRESHOLD = 5.0        # "saturation" if |signal| > this

# CSV columns (update if your CSV uses different names)
COL_AX = "X_norm"
COL_AY = "Y_norm"
COL_AZ = "Z_norm"




def rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    """Compute rolling std (same length as x) using a simple sliding approach."""
    x = x.astype(np.float64)
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    half = win // 2

    for i in range(n):
        s = max(0, i - half)
        e = min(n, i + half + 1)
        out[i] = np.std(x[s:e])
    return out


def run_lengths(binary_arr: np.ndarray) -> np.ndarray:
    """Return lengths of consecutive True runs."""
    if binary_arr.size == 0:
        return np.array([], dtype=int)

    # Find transitions
    d = np.diff(np.concatenate([[0], binary_arr.astype(int), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    if starts.size == 0:
        return np.array([], dtype=int)
    return (ends - starts + 1).astype(int)


def main():
    # -----------------------------
    # Load CSV
    # -----------------------------
    df = pd.read_csv(CSV_PATH)

    for c in [COL_AX, COL_AY, COL_AZ]:
        if c not in df.columns:
            raise ValueError(
                f"Column '{c}' not found in CSV. موجودها: {list(df.columns)}"
            )

    ax = df[COL_AX].to_numpy(dtype=np.float32)
    ay = df[COL_AY].to_numpy(dtype=np.float32)
    az = df[COL_AZ].to_numpy(dtype=np.float32)

    # Magnitude is robust for quality checks
    mag = np.sqrt(ax**2 + ay**2 + az**2)

    n = len(mag)
    if n < WIN_LEN:
        raise ValueError(f"Signal length ({n}) is smaller than WIN_LEN ({WIN_LEN}).")

    # -----------------------------
    # Build "SignalMask-like" bad mask
    # -----------------------------
    is_nan = np.isnan(mag)
    rstd = rolling_std(mag, FLAT_WIN)
    is_flat = rstd < FLAT_STD_EPS
    is_big = np.abs(mag) > SAT_THRESHOLD

    bad = is_nan | is_flat | is_big  # boolean mask same length as signal

    # -----------------------------
    # Window features
    # -----------------------------
    num_wins = ((n - WIN_LEN) // HOP) + 1

    rows = []
    for wid in range(num_wins):
        s = wid * HOP
        e = s + WIN_LEN

        w_bad = bad[s:e]

        ratio = float(np.mean(w_bad))
        runs = run_lengths(w_bad)

        bad_segments = int(len(runs))
        max_bad_run = int(runs.max()) if runs.size else 0
        mean_bad_run = float(runs.mean()) if runs.size else 0.0

        rows.append(
            {
                "window_id": wid + 1,
                "bad_ratio": ratio,
                "bad_segments": bad_segments,
                "max_bad_run": max_bad_run,
                "mean_bad_run": mean_bad_run,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)

    print("✅ Saved SignalMask-style window features to:")
    print("   ", OUT_PATH)
    print("\nPreview:")
    print(out_df.head())


if __name__ == "__main__":
    main()
