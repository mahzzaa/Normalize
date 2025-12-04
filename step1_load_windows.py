# step1_load_windows.py
import json
import os
import numpy as np

# Build a path to the JSON file in the same folder as this script
BASE_DIR = os.path.dirname(__file__)
json_path = os.path.join(BASE_DIR, "sensor_data_14th_may_1600_windows.json")

# Open and load the JSON file
with open(json_path, "r") as f:
    payload = json.load(f)

# Convert "windows" field to a NumPy array
X = np.array(payload["windows"], dtype="float32")  # shape: (num_windows, seq_len, 3)

print("X shape:", X.shape)
print("Example window shape:", X[0].shape)  # should be (seq_len, 3)
print("First row of first window:", X[0][0])
