from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter

from mlp import SimpleMLP

# tensorboard 记录的文件夹名称
g_run_name = "04"

# 超参数
g_num_epochs = 128
g_lr = 0.001
g_batch_size = 256
g_use_cuda = 0
g_dropout = 0.5

# g_hidden_dim = 16
# g_hidden_num = 2
g_early_stop_delta = 1e-4
g_early_stop_patience = 10


def main() -> None:
    print("\n======== 读取处理后的数据")
    base_dir = Path(__file__).resolve().parent
    df_train = pd.read_csv(base_dir / "pre_processed_dataset/dataset/train_processed_v2.csv")
    df_val = pd.read_csv(base_dir / "pre_processed_dataset/dataset/val_processed_v2.csv")

    drop_cols = [c for c in ["Transported", "PassengerId"] if c in df_train.columns]
    df_train_features = df_train.drop(drop_cols, axis=1)
    df_train_target = df_train["Transported"]
    df_val_features = df_val.drop(drop_cols, axis=1)
    df_val_target = df_val["Transported"]
    
    n_train = df_train.shape[0]
    n_val = df_val.shape[0]

    device = torch.device("cuda" if g_use_cuda and torch.cuda.is_available() else "cpu")
    print(f"\n======== 将数据转换为 PyTorch Tensor (device={device})")
    X_train = torch.tensor(df_train_features.values, dtype=torch.float32)
    y_train = torch.tensor(df_train_target.values, dtype=torch.float32).reshape(-1, 1)
    X_val = torch.tensor(df_val_features.values, dtype=torch.float32)
    y_val = torch.tensor(df_val_target.values, dtype=torch.float32).reshape(-1, 1)
    print(X_train.shape)
    print(y_train.shape)

    # create dataloader 
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=g_batch_size, shuffle=True)
    
    # define model
    model = SimpleMLP(dropout=g_dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=g_lr)
    loss = nn.BCELoss()

    # training
    writer = SummaryWriter(f'runs/{g_run_name}')
    prev_epoch_loss = None
    stable_epochs = 0
    total_start = time.perf_counter()
    for epoch in range(g_num_epochs):
        model.train()
        epoch_start = time.perf_counter()
        epoch_loss = 0
        correct_num = 0
        step = 0
        for X_bacth, y_batch in train_loader:
            X_bacth = X_bacth.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_bacth)
            
            correct_num += torch.sum((y_pred > 0.5) == y_batch).item()
            
            l = loss(y_pred, y_batch)
            epoch_loss += l.item()
            
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            
            step += 1

        if step > 0:
            epoch_loss = epoch_loss / step
        
        model.eval()
        with torch.no_grad():
            X_val_device = X_val.to(device)
            y_val_device = y_val.to(device)
            y_val_pred = model(X_val_device)
            val_correct_num = torch.sum((y_val_pred > 0.5) == y_val_device).item()
            val_accuracy = val_correct_num / n_val
        
        train_accuracy = correct_num / n_train
        epoch_time = time.perf_counter() - epoch_start
        print(
            f'Epoch: {epoch}, Train Loss: {epoch_loss:.4f}, '
            f'Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, '
            f'Time: {epoch_time:.2f}s'
        )
        writer.add_scalar('train/accuracy', train_accuracy, epoch)
        writer.add_scalar('train/loss', epoch_loss, epoch)
        writer.add_scalar('val/accuracy', val_accuracy, epoch)

        if prev_epoch_loss is not None and abs(prev_epoch_loss - epoch_loss) < g_early_stop_delta:
            stable_epochs += 1
        else:
            stable_epochs = 0
        prev_epoch_loss = epoch_loss

        if stable_epochs >= g_early_stop_patience:
            print(f"early stop: loss change < {g_early_stop_delta} for {g_early_stop_patience} epochs")
            break



    total_time = time.perf_counter() - total_start
    print(f"training finished in {total_time:.2f}s")

    # save model
    model_dir = base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "mlp.pt"
    torch.save(model.state_dict(), model_path)
    print(f"saved model to {model_path}")


if __name__ == "__main__":
    main()
