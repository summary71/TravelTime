# train_linreg_all.py
import os, argparse, json
import numpy as np
import pandas as pd
import pickle   # using pickle to avoid extra dependencies

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

from vrp_utils import GetModelInfo   # to get model_input_mask

# ---------------------------------------------------------------------
# Settings & Argparse
parser = argparse.ArgumentParser(description="Train simple Linear Regression on full (train + val) dataset.")
parser.add_argument("-nd", "--numdata", type=int, default=-1)
parser.add_argument("-mt", "--modeltype", type=int, default=7)
parser.add_argument("-ms", "--modelsize", type=int, default=0)
parser.add_argument("-bn", "--batchnorm", type=int, default=0)   # kept for consistency
parser.add_argument("-dd", "--datadir", type=str, default="VRPdata")
parser.add_argument("-in", "--inputnorm", type=int, default=0)
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

train_prefix = f"{model_type}_{model_size}_{args.batchnorm}_{data_directory}_linreg_all_in{args.inputnorm}"
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
# Simple Linear Regression with scaling
linreg = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("lr", LinearRegression())
])

linreg.fit(X_all, y_all)
print("[LinearRegression] Model trained on combined train + validation data.")

# ---------------------------------------------------------------------
# Compute MAE and RMSE on the training+validation set
y_all_pred = linreg.predict(X_all)
mae = mean_absolute_error(y_all, y_all_pred)
rmse = sqrt(mean_squared_error(y_all, y_all_pred))
print(f"[LinearRegression] MAE on train+val set : {mae:.6f}")
print(f"[LinearRegression] RMSE on train+val set: {rmse:.6f}")

# ---------------------------------------------------------------------
# Save model
model_path = mydrive + model_saved + f"linreg_{train_prefix}.pkl"
with open(model_path, "wb") as f:
    pickle.dump(linreg, f)
print(f"[LinearRegression] Model saved to {model_path}")
