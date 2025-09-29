# eval_model.py

# =============================
# 1. Imports
# =============================
import os, glob, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import time

from vrp_utils import load_data, split_data, unseen_testdata, split_into_groups, GetModelInfo, get_data
from vrp_model import Net, MyDataset
import json
import glob

# =============================
# 2. Settings & Argparse
# =============================
parser = argparse.ArgumentParser(description="Evaluate VRP src-dst pair driving time estimation.")
parser.add_argument("-mt", "--modeltype", type=int, default=15)
parser.add_argument("-ms", "--modelsize", type=int, default=0)
parser.add_argument("-bn", "--batchnorm", type=int, default=0)
parser.add_argument("-dd", "--datadir", type=str, default="VRPdataSmall")
parser.add_argument("-loss", "--loss", type=str, default="mse")
parser.add_argument("-in", "--inputnorm", type=int, default=0)
args = parser.parse_args()

model_type = args.modeltype
model_size = args.modelsize
loss_type = args.loss # mse, mae
batch_norm = args.batchnorm
data_directory = args.datadir


# now get the model
model_input_mask, small_layer_size, large_layer_size = GetModelInfo(model_type, model_size)
numinputval = sum(model_input_mask)
print("Input mask: ", model_type, model_input_mask, numinputval)

train_prefix = f"{model_type}_{model_size}_{batch_norm}_{data_directory}_{loss_type}_in{args.inputnorm}"
data_prefix = f"{data_directory}_in{args.inputnorm}"
model_saved = "modelsaved/"

# 폰트 크기
plt.rcParams.update({
    'font.size': 14
})

# =============================
# 4. Main Program
# =============================
if __name__ == "__main__":
    mydrive = "./"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(numinputval, small_layer_size, large_layer_size, batch_norm).to(device)
    model.load_state_dict(torch.load(mydrive + model_saved + f"best_model_{train_prefix}.pth", weights_only=True))
    print("Model loaded.")

    json_files = glob.glob(f"../{data_directory}/*.json")
  
    # --- Batch Data Loading ---
    all_data = []
    print("Loading data from all JSON files...")
    for fname in json_files:
        route_data = get_data(fname)
        all_data.extend(route_data)
    print(f"Loaded a total of {len(all_data)} segments from {len(json_files)} files.")

    if all_data:
        # --- Create single Test Set ---
        print("\nCreating a single test set for all OD pairs...")
        df_all_data = pd.DataFrame(all_data)
        all_data_np = df_all_data.values
        model_input_mask_np = np.array(model_input_mask)

        X_test = torch.from_numpy(all_data_np[:, model_input_mask_np]).float()
        
        print(f"Created X_test with shape: {X_test.shape}")

        model.eval()
        with torch.no_grad():
            X_test = X_test.to(device)
            
            start_time = time.time()
            predicted_durations = model(X_test).view(-1)
            end_time = time.time()

            elapsed_time = end_time - start_time
            num_predictions = len(predicted_durations)
            avg_time_per_prediction = elapsed_time / num_predictions if num_predictions > 0 else 0
            
            print("\n--- Prediction Time Complexity ---")
            print(f"Total OD pairs predicted: {num_predictions}")
            print(f"Total prediction time: {elapsed_time:.6f} seconds")
            print(f"Average time per prediction: {avg_time_per_prediction * 1e6:.4f} microseconds")

    else:
        print("No data loaded.")

    
