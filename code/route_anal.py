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

from vrp_utils import load_data, split_data, unseen_testdata, split_into_groups, GetModelInfo, get_data
from vrp_model import Net, MyDataset
import json
import glob

# =============================
# 2. Settings & Argparse
# =============================
parser = argparse.ArgumentParser(description="Evaluate VRP src-dst pair driving time estimation.")
parser.add_argument("-mt", "--modeltype", type=int, default=7)
parser.add_argument("-ms", "--modelsize", type=int, default=0)
parser.add_argument("-bn", "--batchnorm", type=int, default=0)
parser.add_argument("-dd", "--datadir", type=str, default="VRPdataSmall")
parser.add_argument("-loss", "--loss", type=str, default="mse")
parser.add_argument("-routecnt", "--routecnt", type=int, default=1000)
parser.add_argument("-in", "--inputnorm", type=int, default=0)
args = parser.parse_args()

model_type = args.modeltype
model_size = args.modelsize
loss_type = args.loss # mse, mae
batch_norm = args.batchnorm
data_directory = args.datadir

route_count = args.routecnt # how many routes to test

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
    model_path = mydrive + model_saved + f"best_model_{train_prefix}.pth"
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print("Model loaded.")

    json_files = glob.glob(f"../{data_directory}/*.json")
    # Sort files by name (which is unix timestamp), descending to get the latest
    json_files.sort(key=lambda x: os.path.basename(x), reverse=True)
    # Select the latest 'route_count' files
    json_files = json_files[:route_count]
    print(f"Selected {len(json_files)} most recent files to process.")

    route_results = []
    model.eval()  # Set model to evaluation mode once before the loop

    for i, fname in enumerate(json_files):
        print(f"\n--- Processing Route {i+1}/{len(json_files)}: {os.path.basename(fname)} ---")
        route_data = get_data(fname, isInputNorm=args.inputnorm)

        if not route_data:
            print("  No valid data points in this file. Skipping.")
            continue

        # Convert to DataFrame and then to numpy for robust slicing
        df_route_data = pd.DataFrame(route_data)
        route_data_np = df_route_data.values
        model_input_mask_np = np.array(model_input_mask)

        # Create X_test and y_test for THIS route
        X_test = torch.from_numpy(route_data_np[:, model_input_mask_np]).float().to(device)
        y_test = torch.from_numpy(route_data_np[:, -1]).float()

        # Prediction for THIS route
        with torch.no_grad():
            predicted_durations = model(X_test).view(-1).cpu()

        total_predicted_time = torch.sum(predicted_durations).item()
        total_actual_time = torch.sum(y_test).item()

        if total_actual_time == 0:
            print("  Actual time is zero, skipping this route.")
            continue

        difference = total_predicted_time - total_actual_time
        relative_difference = abs(difference / total_actual_time)

        print(f"  Predicted Time: {total_predicted_time:,.2f} sec")
        print(f"  Actual Time:    {total_actual_time:,.2f} sec")
        print(f"  Difference:     {difference:,.2f} sec ({relative_difference:.2f}%)")

        route_results.append({
            'file': os.path.basename(fname),
            'predictions': total_predicted_time,
            'actual_values': total_actual_time,
            'diffs': difference,
            'reldiffs': relative_difference
        })

    # --- Final Summary ---
    print("\n\n--- Overall Summary ---")
    if route_results:
        # Ensure the results directory exists
        os.makedirs("results", exist_ok=True)
        
        df_results = pd.DataFrame(route_results)
        avg_rel_diff = df_results['reldiffs'].mean()
        abs_rel_diff = df_results['reldiffs'].abs().mean()
        
        total_predicted = df_results['predictions'].sum()
        total_actual = df_results['actual_values'].sum()
        overall_rel_diff = ((total_predicted - total_actual) / total_actual) * 100 if total_actual > 0 else 0

        print(f"Processed {len(route_results)} routes.")
        print(f"Average Relative Difference per route: {avg_rel_diff:.2f}%")
        print(f"Average Absolute Relative Difference per route: {abs_rel_diff:.2f}%")
        print("-" * 20)
        print(f"Total Predicted Time (all routes): {total_predicted:,.2f} sec")
        print(f"Total Actual Time (all routes):    {total_actual:,.2f} sec")
        print(f"Overall Relative Difference:       {overall_rel_diff:.2f}%")
        
        # Save the detailed results to a CSV file
        results_path = f"results/{train_prefix}_route_analysis_summary.csv"
        df_results.to_csv(results_path, index=False)
        print(f"\nRoute analysis summary saved to {results_path}")

        # --- CDF Plot ---
        # Ensure the graphs directory exists
        os.makedirs("graphs", exist_ok=True)
        
        # Data for CDF: absolute relative difference for each route
        abs_rel_diff_data = df_results['reldiffs'].abs()
        
        # Sort data for plotting
        sorted_data = np.sort(abs_rel_diff_data)
        
        # Calculate cumulative probabilities
        yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        plt.figure(figsize=(10, 6))
        plt.plot(sorted_data, yvals)
        plt.title('Cumulative Distribution of Absolute Relative Difference per Route')
        plt.xlabel('Absolute Relative Difference (|Predicted - Actual| / Actual)')
        plt.ylabel('Cumulative Probability')
        plt.xlim([0, 1])
        plt.ylim([0, 1.01])
        plt.grid(True)
        
        # Save the plot
        plot_path = f"graphs/{train_prefix}_cdf_abs_rel_diff.pdf"
        plt.savefig(plot_path)
        plt.close()
        print(f"CDF plot saved to {plot_path}")

    else:
        print("No routes were processed.")

