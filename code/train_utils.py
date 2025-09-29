# train_utils.py
# ------------------------
# 학습 함수 정의:
# - train_model: 학습 loop, early stopping 포함
# ------------------------

import torch
import numpy as np
from datetime import datetime

def validation_interval(epoch, num_epochs):
    """
    Adaptive validation interval:
    - 초반에는 interval 크게 (validation 자주 X)
    - 후반에는 interval 작게 (validation 자주 O)
    """
    min_interval = 1
    max_interval = 10
    decay_factor = 5.0  # tuning 가능 (클수록 빠르게 감소)
    progress = epoch / num_epochs

    interval = max_interval * np.exp(-decay_factor * progress)
    return max(int(interval), min_interval)

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion,
                num_epochs, device, patience=10, min_delta=0.0, model_saved_path="./", train_prefix=""):
    model.train()
    print("Training started:", datetime.now())

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # ---- Adaptive Validation ----
        interval = validation_interval(epoch, num_epochs)
        if epoch % interval == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    val_outputs = model(val_inputs)
                    val_loss += criterion(val_outputs, val_targets).item()

            val_loss /= len(val_loader)
            print(f"Epoch {epoch}/{num_epochs}, Train Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f} (Interval: {interval})")

            # Early stopping logic
            if val_loss + min_delta < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                print(f" --> New best val_loss: {best_val_loss:.4f}")

                # Save best model
                torch.save(model.state_dict(), model_saved_path + f"best_model_{train_prefix}.pth")
                print(f" --> Best model saved to best_model_{train_prefix}.pth")
            else:
                epochs_without_improvement += 1
                print(f" --> No improvement ({epochs_without_improvement}/{patience})")
                if epochs_without_improvement >= patience:
                    print("Early stopping triggered!")
                    break

            # Scheduler update
            scheduler.step(val_loss)

    print("Training finished:", datetime.now())
