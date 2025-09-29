# train_model.py
import os, glob, argparse, torch, json
import pandas as pd
import numpy as np
import math

from torch import nn, optim
from torch.utils.data import DataLoader
import torch.onnx

from vrp_model import Net, MyDataset
from vrp_utils import GetModelInfo
from train_utils import train_model

# Main
mydrive = "./"

# Settings & Argparse
parser = argparse.ArgumentParser(description="Train VRP src-dst pair driving time estimation.")
parser.add_argument("-ne", "--numepoch", type=int, default=5000)
parser.add_argument("-nd", "--numdata", type=int, default=-1)
parser.add_argument("-mt", "--modeltype", type=int, default=7)
parser.add_argument("-ms", "--modelsize", type=int, default=0)
parser.add_argument("-bn", "--batchnorm", type=int, default=0)
parser.add_argument("-dd", "--datadir", type=str, default="VRPdata")
parser.add_argument("-loss", "--loss", type=str, default="mse")
parser.add_argument("-in", "--inputnorm", type=int, default=0)
args = parser.parse_args()

num_epochs = args.numepoch
num_data = args.numdata

model_type = args.modeltype # 1: coordis, 2: geodesic , 3: coordis + geodesic
model_size = args.modelsize # 0: small, 1: large
loss_type = args.loss # mse, mae

model_input_mask, small_layer_size, large_layer_size = GetModelInfo(model_type, model_size)
numinputval = sum(model_input_mask) # input data length
print(f"Input mask: model_type: {model_type}, model_input_mask: {model_input_mask}, numinputval: {numinputval}")

batch_norm = args.batchnorm
data_directory = args.datadir
train_prefix = f"{model_type}_{model_size}_{batch_norm}_{data_directory}_{loss_type}_in{args.inputnorm}"
data_prefix = f"{data_directory}_in{args.inputnorm}"
model_saved = "modelsaved/"
# 디렉토리 없으면 생성
os.makedirs(model_saved, exist_ok=True)

# Load split data
with open(f"{mydrive}split_data_{data_directory}/{data_prefix}_traindata.json", 'r') as f:
    traindata = np.array(json.load(f))
with open(f"{mydrive}split_data_{data_directory}/{data_prefix}_valdata.json", 'r') as f:
    valdata = np.array(json.load(f))

df_train = pd.DataFrame(traindata)
X_train = torch.from_numpy(df_train.iloc[:, model_input_mask].values).float()
y_train = torch.from_numpy(df_train.iloc[:, -1].values).float()
train_dataset = MyDataset(X_train, y_train)
# 원래 batch size 후보
initial_batch_size = X_train.shape[0] // 5
# 2의 거듭제곱 중 가장 큰 값으로 조정
batch_size = 2 ** int(math.floor(math.log2(initial_batch_size)))
batch_size = 32

print(f"Batch size: {batch_size}, memory usage: {X_train.nbytes / 1024 / 1024:.2f} MB")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

df_val = pd.DataFrame(valdata)
X_val = torch.from_numpy(df_val.iloc[:, model_input_mask].values).float()
y_val = torch.from_numpy(df_val.iloc[:, -1].values).float()
val_dataset = MyDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

model = Net(numinputval, small_layer_size, large_layer_size, batch_norm).to(device)

base_lr = 1e-3
scaling_factor = (batch_size / 32) ** 0.5
new_lr = base_lr * scaling_factor
# 상한값 적용
lr_cap = 1e-2  # 또는 3e-2 정도까지 실험 가능
new_lr = min(new_lr, lr_cap)
new_lr = base_lr
print(f"Learning rate: {new_lr}")

optimizer = optim.Adam(model.parameters(), lr=new_lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

if args.loss == "mse":
    criterion = nn.MSELoss()
elif args.loss == "mae":
    criterion = nn.SmoothL1Loss()

train_model(model, train_loader, val_loader, optimizer, scheduler, criterion,
            num_epochs, device, patience=20, min_delta=0.0,
            model_saved_path=mydrive + model_saved, train_prefix=train_prefix)

torch.save(model.state_dict(), mydrive + model_saved + f"model_{train_prefix}.pth")
print("Model saved.")

dummy_input = torch.randn(1, numinputval, device=device)
onnx_path = mydrive + model_saved + f"model_{train_prefix}.onnx"
torch.onnx.export(model, dummy_input, onnx_path,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11)
print(f"Model exported to {onnx_path}")

