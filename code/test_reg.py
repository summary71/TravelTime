# eval_linreg.py

# =============================
# 1. Imports
# =============================
import os, argparse, json, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

from sklearn.metrics import mean_absolute_error, mean_squared_error

from vrp_utils import unseen_testdata, split_into_groups, GetModelInfo

# =============================
# 2. Settings & Argparse
# =============================
parser = argparse.ArgumentParser(description="Evaluate Linear Regression VRP travel-time estimator.")
parser.add_argument("-nd", "--numdata", type=int, default=-1)
parser.add_argument("-mt", "--modeltype", type=int, default=7)
parser.add_argument("-ms", "--modelsize", type=int, default=0)
parser.add_argument("-bn", "--batchnorm", type=int, default=0)   # kept for prefix compatibility
parser.add_argument("-dd", "--datadir", type=str, default="VRPdata")
parser.add_argument("-in", "--inputnorm", type=int, default=0)
parser.add_argument("--modeltag", type=str, default="linreg_all",
                    help="Suffix tag used when training (e.g., 'linreg_all').")
args = parser.parse_args()

num_data   = args.numdata
model_type = args.modeltype
model_size = args.modelsize

# Derive input mask and sizes just like in training
model_input_mask, small_layer_size, large_layer_size = GetModelInfo(model_type, model_size)
numinputval = sum(model_input_mask)
print("Input mask:", model_type, model_input_mask, numinputval)

batch_norm = args.batchnorm
data_directory = args.datadir

# Match the training prefix logic used in train_linreg_all.py
train_prefix = f"{model_type}_{model_size}_{batch_norm}_{data_directory}_{args.modeltag}_in{args.inputnorm}"
data_prefix  = f"{data_directory}_in{args.inputnorm}"
model_saved  = "modelsaved/"

# Fonts
plt.rcParams.update({'font.size': 14})

# =============================
# 3. Evaluation helpers
# =============================
def evaluate_array(model, X_np, y_np, name_suffix):
    """Compute predictions, diffs, relative diffs, MAE, RMSE."""
    y_pred = model.predict(X_np).astype(np.float64)
    y_true = y_np.astype(np.float64)

    diffs = y_pred - y_true
    # avoid div by zero
    reldiffs = np.abs(diffs) / np.maximum(np.abs(y_true), 1e-6)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mean_reldiff = np.mean(reldiffs)
    percentiles_reldiff = np.percentile(reldiffs, [10, 50, 90])

    print(f"Evaluation {name_suffix}:")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Mean relative diff: {mean_reldiff:.6f}")
    print(f"  Rel diff percentiles (10th, median, 90th): "
          f"{percentiles_reldiff[0]:.6f}, {percentiles_reldiff[1]:.6f}, {percentiles_reldiff[2]:.6f}")

    return y_true, y_pred, diffs, reldiffs, mae, rmse, mean_reldiff, percentiles_reldiff

def evaluate_and_save(data_array, name_suffix, model, mydrive, input_mask):
    df_test = pd.DataFrame(data_array)
    X_test = df_test.iloc[:, input_mask].values.astype(np.float32)
    y_test = df_test.iloc[:, -1].values.astype(np.float32)

    (actual_values, predictions, diffs, reldiffs,
     mae, rmse, mean_reldiff, percentiles_reldiff) = evaluate_array(model, X_test, y_test, name_suffix)

    # Save per-sample results
    df_eval = pd.DataFrame({
        'actual_values': actual_values,
        'predictions': predictions,
        'diffs': diffs,
        'reldiffs': reldiffs
    })
    results_dir = f"{mydrive}results/"
    os.makedirs(results_dir, exist_ok=True)
    result_file = f"{results_dir}/{train_prefix}_evaluation_results_{name_suffix}.csv"
    df_eval.to_csv(result_file, index=False)
    print(f"Evaluation results saved to {result_file}")

    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_values, predictions, alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted Values ({name_suffix})')
    plt.grid(True)
    graphs_dir = f"{mydrive}graphs/"
    os.makedirs(graphs_dir, exist_ok=True)
    graph_file = f"{graphs_dir}/{train_prefix}_actual_vs_predicted_{name_suffix}.pdf"
    plt.savefig(graph_file)
    print(f"Graph saved to {graph_file}")
    plt.close()

    return mae, rmse, mean_reldiff, percentiles_reldiff, reldiffs

# =============================
# 4. Main Program
# =============================
if __name__ == "__main__":
    mydrive = "./"

    # Load splits
    with open(f"{mydrive}split_data_{data_directory}/{data_prefix}_valdata.json", 'r') as f:
        valdata = np.array(json.load(f))
    with open(f"{mydrive}split_data_{data_directory}/{data_prefix}_traindata.json", 'r') as f:
        traindata = np.array(json.load(f))
    with open(f"{mydrive}split_data_{data_directory}/{data_prefix}_testdata.json", 'r') as f:
        testdata = np.array(json.load(f))

    # Build unseen subsets and group by your distance boundaries
    unseen_test_data = unseen_testdata(traindata, valdata, testdata)
    group_boundary = [5, 10, 20]
    groups   = split_into_groups(unseen_test_data, group_boundary)
    orggroups = split_into_groups(testdata, group_boundary)

    # Load the saved Linear Regression model
    model_path = mydrive + model_saved + f"linreg_{train_prefix}.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find saved model at {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded model from {model_path}")

    # Assemble test sets to evaluate
    test_sets = [
        (testdata, "testdata"),
        (unseen_test_data, "unseen_data")
    ]
    for i, group in enumerate(groups):
        if i == 0:
            name_suffix = f"[0, {group_boundary[i]}) km"
        elif i < len(group_boundary):
            name_suffix = f"[{group_boundary[i - 1]}, {group_boundary[i]}) km"
        else:
            name_suffix = f"[{group_boundary[-1]}, ∞) km"
        test_sets.append((group, name_suffix))

    for i, group in enumerate(orggroups):
        if i == 0:
            name_suffix = f"seen_[0, {group_boundary[i]}) km"
        elif i < len(group_boundary):
            name_suffix = f"seen_[{group_boundary[i - 1]}, {group_boundary[i]}) km"
        else:
            name_suffix = f"seen_[{group_boundary[-1]}, ∞) km"
        test_sets.append((group, name_suffix))

    # Evaluate and gather results for CDF plot
    results_summary = []
    for data_array, name_suffix in test_sets:
        mae, rmse, mean_reldiff, percentiles_reldiff, reldiffs = evaluate_and_save(
            data_array, name_suffix, model, mydrive, model_input_mask
        )
        # Print MAE and RMSE explicitly (already printed inside evaluate_array, but reiterated here for clarity)
        print(f"[{name_suffix}] MAE: {mae:.6f} | RMSE: {rmse:.6f}")
        results_summary.append((name_suffix, mean_reldiff, percentiles_reldiff, reldiffs))

    # Plot cumulative distribution of relative errors for all groups
    plt.figure(figsize=(10, 6))
    for name_suffix, mean_reldiff, percentiles_reldiff, reldiffs in results_summary:
        sorted_reldiffs = np.sort(reldiffs)
        if len(sorted_reldiffs) == 0:
            continue
        cumulative_probs = np.arange(1, len(sorted_reldiffs) + 1) / len(sorted_reldiffs)
        plt.plot(sorted_reldiffs, cumulative_probs, label=name_suffix)

    plt.xlabel("Relative Errors", fontsize=14)
    plt.ylabel("Cumulative Probability", fontsize=14)
    plt.title("Cumulative Distribution of Relative Errors", fontsize=16)
    plt.xlim(0, 1)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    os.makedirs(f"{mydrive}graphs/", exist_ok=True)
    plt.savefig(f"{mydrive}graphs/{train_prefix}_rel_diff_cumulative_distribution.pdf")
    plt.close()
