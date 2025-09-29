# vrp_model.py
# ------------------------
# 모델 정의:
# - Net: Fully connected DNN
# - MyDataset: torch Dataset wrapper
# ------------------------

import torch
from torch import nn
from torch.utils.data import Dataset



class Net(nn.Module):
    def __init__(self, idim, smalllayer=64, largelayer=128, batchnorm=1):
        super(Net, self).__init__()
        # Layer 구성
        self.fc1 = nn.Linear(idim, smalllayer)
        self.bn1 = nn.BatchNorm1d(smalllayer) if batchnorm == 1 else None
        self.fc2 = nn.Linear(smalllayer, largelayer)
        self.bn2 = nn.BatchNorm1d(largelayer) if batchnorm == 1 else None
        self.fc3 = nn.Linear(largelayer, smalllayer)
        self.bn3 = nn.BatchNorm1d(smalllayer) if batchnorm == 1 else None
        self.fc4 = nn.Linear(smalllayer, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        if self.bn1 is not None:
            x = self.bn1(x)
        x = torch.relu(self.fc2(x))
        if self.bn2 is not None:
            x = self.bn2(x)
        x = torch.relu(self.fc3(x))
        if self.bn3 is not None:
            x = self.bn3(x)
        return self.fc4(x)

class MyDataset(Dataset):
    def __init__(self, data, targets):
        # Dataset 구성
        self.data = data
        if targets.ndim == 1:
            self.targets = targets.unsqueeze(1)
        else:
            self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

