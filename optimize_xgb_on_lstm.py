# optimize_xgb_on_lstm.py

import os, json
import numpy as np
from tensorflow.keras.models import load_model, Model
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(__file__)

JSON_PATH = os.path.join(BASE_DIR, "sensor_data_14th_may_1600_windows_with_labels.json")
MODEL_PATH = os.path.join(BASE_DIR, "gait_lstm_autoencoder.h5")

# 1) Load labeled data
with open(JSON_PATH, "r") as f:
    payload = json.load(f)

X_all = np.array(payload["windows"], dtype="float32")
y_all = np.array(payload["labels"], dtype="int32")

print("Loaded labeled dataset:", X_all.shape, y_all.shape)

# 2) Load trained AE (IMPORTANT: compile=False avoids metric deserialization issues)
autoencoder = load_model(MODEL_PATH, compile=False)
print("Loaded autoencoder from:", MODEL_PATH)

# 3) Build encoder model (input -> encoder output)
encoder_layer = autoencoder.get_layer("encoder_lstm")
encoder = Model(inputs=autoencoder.input, outputs=encoder_layer.output)

# 4) Latent features
Z_all = encoder.predict(X_all, verbose=0)
print("Latent features shape (Z_all):", Z_all.shape)

# 5) XGBoost hyperparameter search (optimize F1, not accuracy)
param_dist = {
    "n_estimators": [100, 200, 400, 600],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [0.1, 1.0, 5.0, 10.0],
    "min_child_weight": [1, 3, 5, 10],
    "gamma": [0, 0.1, 0.3, 0.5],
}

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=40,
    scoring=make_scorer(f1_score),
    cv=5,
    verbose=2,
    n_jobs=-1
)

search.fit(Z_all, y_all)

print("\nBest parameters:", search.best_params_)
print("Best CV F1:", search.best_score_)
