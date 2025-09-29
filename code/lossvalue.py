import re
import sys
import matplotlib.pyplot as plt
import os
import argparse

# Set up command line argument parser
parser = argparse.ArgumentParser(description='Generate loss comparison graphs from log files')
parser.add_argument('log_file', help='Path to the log file to analyze')
parser.add_argument('--inno-flag', '-i', type=int, choices=[0, 1], default=0,
                   help='InNo flag: 0 for standard comparison, 1 for InNo comparison (default: 0)')
parser.add_argument('--output', '-o', default='lossvalue.pdf',
                   help='Output file name for the graph (default: lossvalue.pdf)')

# Parse command line arguments
args = parser.parse_args()

log_file = args.log_file
inno_flag = args.inno_flag
output_file = args.output

try:
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
except FileNotFoundError:
    print(f"File not found: {log_file}")
    sys.exit(1)

results = []
current_val_losses = []
current_train_losses = []
current_onnx = None

for line in lines:
    if "Input mask:" in line:
        current_val_losses = []
        current_train_losses = []
        current_onnx = None

    match_val_loss = re.search(r"Validation Loss:\s*([\d.]+)", line)
    if match_val_loss:
        val_loss = float(match_val_loss.group(1))
        current_val_losses.append(val_loss)

    match_train_loss = re.search(r"Train Loss:\s*([\d.]+)", line)
    if match_train_loss:
        train_loss = float(match_train_loss.group(1))
        current_train_losses.append(train_loss)

    if "Model exported to" in line:
        match_onnx = re.search(r'Model exported to\s+(.+\.onnx)', line)
        if match_onnx and current_val_losses:
            current_onnx = match_onnx.group(1).strip()
            best_val_loss = min(current_val_losses)
            # Find the training loss corresponding to the best validation loss
            best_val_index = current_val_losses.index(best_val_loss)
            if best_val_index < len(current_train_losses):
                corresponding_train_loss = current_train_losses[best_val_index]
            else:
                # If no corresponding train loss, use the last available one
                corresponding_train_loss = current_train_losses[-1] if current_train_losses else 0
            results.append((current_onnx, best_val_loss, corresponding_train_loss))

# Print results
for onnx_file, val_loss, train_loss in results:
    print(f"ONNX File: {onnx_file}")
    print(f"  Min Validation Loss: {val_loss:.4f}")
    print(f"  Corresponding Train Loss: {train_loss:.4f}\n")


if inno_flag == 1:
    # Filter for specific ONNX files
    target_files = [
        "./modelsaved/model_1_0_0_VRPdata_mse_in0.onnx",
        "./modelsaved/model_1_0_0_VRPdata_mse_in1.onnx",
        "./modelsaved/model_7_0_0_VRPdata_mse_in0.onnx",
        "./modelsaved/model_7_0_0_VRPdata_mse_in1.onnx"
    ]


    # Custom labels for the x-axis in the same order as target_files
    custom_labels = [
        "w/o InNo\ninput conf 1",
        "w/ InNo\ninput conf 1", 
        "w/o InNo\ninput conf 7",
        "w/ InNo\ninput conf 7"
    ]
else:
    # Filter for specific ONNX files
    target_files = [
        "./modelsaved/model_1_0_0_VRPdata_mse_in0.onnx",
        "./modelsaved/model_3_0_0_VRPdata_mse_in0.onnx",
        "./modelsaved/model_5_0_0_VRPdata_mse_in0.onnx",
        "./modelsaved/model_9_0_0_VRPdata_mse_in0.onnx",
        "./modelsaved/model_1_1_0_VRPdata_mse_in0.onnx"
    ]


    # Custom labels for the x-axis in the same order as target_files
    custom_labels = [
        "input conf 1",
        "input conf 3", 
        "input conf 5",
        "input conf 9",
        "input conf 1 (Large Size)"
    ]

filtered_results = []
for onnx_file, val_loss, train_loss in results:
    if onnx_file in target_files:
        filtered_results.append((onnx_file, val_loss, train_loss))

if not filtered_results:
    print("No matching ONNX files found in the results.")
    sys.exit(1)

# Create graphs directory if it doesn't exist
os.makedirs("./graphs", exist_ok=True)

# Prepare data for plotting
file_names = []
val_loss_values = []
train_loss_values = []


for onnx_file, val_loss, train_loss in filtered_results:
    # Find the index of this file in target_files to get the corresponding label
    try:
        index = target_files.index(onnx_file)
        file_names.append(custom_labels[index])
    except ValueError:
        # Fallback to filename if not found
        filename = os.path.basename(onnx_file)
        file_names.append(filename)
    val_loss_values.append(val_loss)
    train_loss_values.append(train_loss)

# Create the grouped bar graph
plt.figure(figsize=(12, 8))

# Set up the positions for grouped bars
x = range(len(file_names))
width = 0.35

# Create bars for training and validation loss (training on the left)
train_bars = plt.bar([i - width/2 for i in x], train_loss_values, width, label='Training Loss', color='#ff7f0e')
val_bars = plt.bar([i + width/2 for i in x], val_loss_values, width, label='Validation Loss', color='#1f77b4')

# Customize the plot
plt.title('Loss Comparison', fontsize=20, fontweight='bold')
plt.xlabel('Input Configurations', fontsize=16)
plt.ylabel('Loss (MSE)', fontsize=16)
plt.xticks(x, file_names, rotation=0, ha='center', fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)

# Format y-axis labels with comma separators (integers only)
import matplotlib.ticker as ticker
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
plt.grid(axis='y', alpha=0.3)

# Add value labels on top of bars
for bar, loss in zip(val_bars, val_loss_values):
    height = bar.get_height()
    # Round to 1 decimal place and add thousands comma separator
    formatted_loss = f'{loss:,.1f}'
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             formatted_loss, ha='center', va='bottom', fontweight='bold', fontsize=12)

for bar, loss in zip(train_bars, train_loss_values):
    height = bar.get_height()
    # Round to 1 decimal place and add thousands comma separator
    formatted_loss = f'{loss:,.1f}'
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             formatted_loss, ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()

# Save the plot
output_path = f"./graphs/{output_file}"
plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
print(f"Bar graph saved to: {output_path}")

# Close the figure to free memory
plt.close()
