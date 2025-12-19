import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "sensor_data_14th_may_1600_normalized.csv")

N = 1000

df = pd.read_csv(CSV_PATH)

x = df["X_norm"].to_numpy()[:N]
y = df["Y_norm"].to_numpy()[:N]
t = np.arange(N)

angle = np.arctan2(y, x)          # radians
sin_a = np.sin(angle)
cos_a = np.cos(angle)

plt.figure()
plt.plot(t, angle, label="angle atan2(Y_norm, X_norm)")
plt.xlabel("Sample index")
plt.ylabel("Radians")
plt.title("Angle over time")
plt.legend()
plt.savefig(os.path.join(BASE_DIR, "angle.png"), dpi=150, bbox_inches="tight")
plt.show()

plt.figure()
plt.plot(t, sin_a, label="sin(angle)")
plt.plot(t, cos_a, label="cos(angle)")
plt.xlabel("Sample index")
plt.ylabel("Value")
plt.title("sin/cos of angle")
plt.legend()
plt.savefig(os.path.join(BASE_DIR, "sin_cos_angle.png"), dpi=150, bbox_inches="tight")
plt.show()

print("Saved: angle.png and sin_cos_angle.png")
