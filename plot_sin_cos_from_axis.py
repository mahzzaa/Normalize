import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "sensor_data_14th_may_1600_normalized.csv")

AXIS = "X_norm"   # change to Y_norm or Z_norm if you want
N = 1000          # how many samples to plot (keep it readable)

df = pd.read_csv(CSV_PATH)

x = df[AXIS].to_numpy()[:N]
t = np.arange(len(x))  # sample index (or use Timestamp if you want)

sin_x = np.sin(x)
cos_x = np.cos(x)

plt.figure()
plt.plot(t, x, label=f"{AXIS}")
plt.plot(t, sin_x, label=f"sin({AXIS})")
plt.plot(t, cos_x, label=f"cos({AXIS})")
plt.xlabel("Sample index")
plt.ylabel("Value")
plt.title(f"Signal + sin/cos transform ({AXIS})")
plt.legend()

out = os.path.join(BASE_DIR, f"sin_cos_{AXIS}.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print("Saved:", out)
