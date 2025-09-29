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

from vrp_utils import load_data, split_data, unseen_testdata, split_into_groups, GetModelInfo
from vrp_model import Net, MyDataset

# =============================
# 2. Settings & Argparse
# =============================
parser = argparse.ArgumentParser(description="Evaluate VRP src-dst pair driving time estimation.")
parser.add_argument("-nd", "--numdata", type=int, default=-1)
parser.add_argument("-mt", "--modeltype", type=int, default=7)
parser.add_argument("-ms", "--modelsize", type=int, default=0)
parser.add_argument("-bn", "--batchnorm", type=int, default=0)
parser.add_argument("-dd", "--datadir", type=str, default="VRPdata")
parser.add_argument("-loss", "--loss", type=str, default="mse")
parser.add_argument("-in", "--inputnorm", type=int, default=0)
args = parser.parse_args()

num_data = args.numdata

model_type = args.modeltype
model_size = args.modelsize
loss_type = args.loss # mse, mae

model_input_mask, small_layer_size, large_layer_size = GetModelInfo(model_type, model_size)
numinputval = sum(model_input_mask)
print("Input mask: ", model_type, model_input_mask, numinputval)

batch_norm = args.batchnorm

data_directory = args.datadir
train_prefix = f"{model_type}_{model_size}_{batch_norm}_{data_directory}_{loss_type}_in{args.inputnorm}"
data_prefix = f"{data_directory}_in{args.inputnorm}"
model_saved = "modelsaved/"

# 폰트 크기
plt.rcParams.update({
    'font.size': 14
})

# =============================
# 3. Evaluation Function
# =============================
def evaluate_model(model, loader, device, name_suffix):
    model.eval()
    predictions, actual_values, diffs, reldiffs = [], [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1)

            outputs = model(inputs).view(-1)

            for i in range(len(labels)):
                if labels[i].item() == 0:
                    continue
                predictions.append(outputs[i].item())
                actual_values.append(labels[i].item())
                diffs.append(outputs[i].item() - labels[i].item())
                reldiff = abs(outputs[i].item() - labels[i].item()) / max(labels[i].item(), 1e-6)
                reldiffs.append(reldiff)

    mean_diff = np.mean(diffs)
    mean_reldiff = np.mean(reldiffs)
    percentiles_reldiff = np.percentile(reldiffs, [10, 50, 90])

    print(f"Evaluation {name_suffix}:")
    print(f"  Mean diff: {mean_diff:.2f} sec")
    print(f"  Mean rel diff: {mean_reldiff:.4f}")
    print(f"  Rel diff percentiles (10th, median, 90th): {percentiles_reldiff[0]:.4f}, {percentiles_reldiff[1]:.4f}, {percentiles_reldiff[2]:.4f}")

    return np.array(actual_values), np.array(predictions), np.array(diffs), np.array(reldiffs), mean_reldiff, percentiles_reldiff

def evaluate_and_save(data_array, name_suffix, model, device, mydrive, input_mask):
    df_test = pd.DataFrame(data_array)
    X_test = torch.from_numpy(df_test.iloc[:, input_mask].values).float()
    y_test = torch.from_numpy(df_test.iloc[:, -1].values).float()
    test_dataset = MyDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    actual_values, predictions, diffs, reldiffs, mean_reldiff, percentiles_reldiff = evaluate_model(model, test_loader, device, name_suffix)

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

    return mean_reldiff, percentiles_reldiff, reldiffs

# =============================
# 4. Main Program
# =============================
if __name__ == "__main__":
    mydrive = "./"
    # Load split testdata
    with open(f"{mydrive}split_data_{data_directory}/{data_prefix}_valdata.json", 'r') as f:
        valdata = np.array(json.load(f))
    with open(f"{mydrive}split_data_{data_directory}/{data_prefix}_traindata.json", 'r') as f:
        traindata = np.array(json.load(f))
    with open(f"{mydrive}split_data_{data_directory}/{data_prefix}_testdata.json", 'r') as f:
        testdata = np.array(json.load(f))

    unseen_test_data = unseen_testdata(traindata, valdata, testdata)
    group_boundary = [5, 10, 20]
    groups = split_into_groups(unseen_test_data, group_boundary)
    orggroups = split_into_groups(testdata, group_boundary)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = Net(numinputval, small_layer_size, large_layer_size, batch_norm).to(device)
    model.load_state_dict(torch.load(mydrive + model_saved + f"best_model_{train_prefix}.pth"))
    print("Model loaded.")

    # Run evaluation on all test sets
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

   
    results_summary = []
    for data_array, name_suffix in test_sets:
        mean_reldiff, percentiles_reldiff, reldiffs = evaluate_and_save(data_array, name_suffix, model, device, mydrive, model_input_mask)
        results_summary.append((name_suffix, mean_reldiff, percentiles_reldiff, reldiffs))

    # Plot cumulative distribution of rel diff per test group
    plt.figure(figsize=(10, 6))
    for name_suffix, mean_reldiff, percentiles_reldiff, reldiffs in results_summary:
        sorted_reldiffs = np.sort(reldiffs)
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
    plt.savefig(f"{mydrive}graphs/{train_prefix}_rel_diff_cumulative_distribution.pdf")
    plt.close()
