# train_gbt_all.py
import os, argparse, json
import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

from vrp_utils import GetModelInfo   # to get model_input_mask

# ---------------------------------------------------------------------
# Settings & Argparse
parser = argparse.ArgumentParser(description="Train Gradient Boosted Trees on full (train + val) dataset.")
parser.add_argument("-nd", "--numdata", type=int, default=-1)
parser.add_argument("-mt", "--modeltype", type=int, default=7)
parser.add_argument("-ms", "--modelsize", type=int, default=0)
parser.add_argument("-bn", "--batchnorm", type=int, default=0)   # kept for prefix compatibility
parser.add_argument("-dd", "--datadir", type=str, default="VRPdata")
parser.add_argument("-in", "--inputnorm", type=int, default=0)
# Optional hyperparameters
parser.add_argument("--n_estimators", type=int, default=500)
parser.add_argument("--learning_rate", type=float, default=0.05)
parser.add_argument("--max_depth", type=int, default=3)
parser.add_argument("--subsample", type=float, default=0.8)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# ---------------------------------------------------------------------
mydrive = "./"

model_type = args.modeltype
model_size = args.modelsize
data_directory = args.datadir

# Get input mask (which columns to use as features)
model_input_mask, _, _ = GetModelInfo(model_type, model_size)
numinputval = sum(model_input_mask)
print(f"Input mask: model_type: {model_type}, model_input_mask: {model_input_mask}, numinputval: {numinputval}")

train_prefix = f"{model_type}_{model_size}_{args.batchnorm}_{data_directory}_gbr_all_in{args.inputnorm}"
data_prefix  = f"{data_directory}_in{args.inputnorm}"
model_saved  = "modelsaved/"
os.makedirs(model_saved, exist_ok=True)

# ---------------------------------------------------------------------
# Load training + validation data and concatenate
with open(f"{mydrive}split_data_{data_directory}/{data_prefix}_traindata.json", 'r') as f:
    traindata = np.array(json.load(f))
with open(f"{mydrive}split_data_{data_directory}/{data_prefix}_valdata.json", 'r') as f:
    valdata = np.array(json.load(f))

all_data = np.vstack([traindata, valdata])
df_all   = pd.DataFrame(all_data)

X_all = df_all.iloc[:, model_input_mask].values.astype(np.float32)
y_all = df_all.iloc[:, -1].values.astype(np.float32)

print(f"Total training samples (train + val): {X_all.shape[0]}")

# ---------------------------------------------------------------------
# Gradient Boosted Trees
gbr = GradientBoostingRegressor(
    loss="squared_error",
    n_estimators=args.n_estimators,
    learning_rate=args.learning_rate,
    max_depth=args.max_depth,
    subsample=args.subsample,
    random_state=args.random_state
)

gbr.fit(X_all, y_all)
print("[GBT] Model trained on combined train + validation data.")

# ---------------------------------------------------------------------
# Print MAE and RMSE on the combined training set (for reference)
y_all_pred = gbr.predict(X_all).astype(np.float64)
mae  = mean_absolute_error(y_all, y_all_pred)
rmse = sqrt(mean_squared_error(y_all, y_all_pred))
print(f"[GBT] MAE on train+val set : {mae:.6f}")
print(f"[GBT] RMSE on train+val set: {rmse:.6f}")

# ---------------------------------------------------------------------
# Save model
model_path = mydrive + model_saved + f"gbr_{train_prefix}.pkl"
with open(model_path, "wb") as f:
    pickle.dump(gbr, f)
print(f"[GBT] Model saved to {model_path}")
