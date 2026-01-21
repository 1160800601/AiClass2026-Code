import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd
import time

from mlp import SimpleMLP
from cnn import SimpleCNN
from preprocess import preprocess
import utils

# TensorBoard log directory name.
# run_name = 'mlp01'
run_name = 'cnn01'

# Hyperparameters.
g_num_epochs = 20
g_lr = 0.01
g_batch_size = 500
g_weight_decay = 1e-4

input_dim = 28 * 28
hidden_dim = 16
hidden_num = 2

# model_flag = 0 # mlp
model_flag = 1 # cnn


def main():
    # Select device.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    
    # Load preprocessed data.
    train_data, train_label, test_data = preprocess('dataset/train1.csv', 'dataset/test.csv')
    val_data, val_label, test_data = preprocess('dataset/val1.csv', 'dataset/test.csv')
    n_train = train_data.shape[0]
    n_val = val_data.shape[0]
    n_test = test_data.shape[0]
    
    X_train = torch.tensor(train_data, dtype=torch.float32)
    y_train = torch.tensor(train_label, dtype=torch.int8).reshape(-1, 1)
    X_val = torch.tensor(val_data, dtype=torch.float32)
    y_val = torch.tensor(val_label, dtype=torch.int8).reshape(-1, 1)
    X_test = torch.tensor(test_data, dtype=torch.float32)
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    print(X_test.shape)
    # Randomly pick images from the training set to visualize.
    utils.draw_imgs(X_train, y_train)
    
    y_train = torch.tensor(np.eye(10)[y_train.reshape(-1)], dtype=torch.float32)
    y_val = torch.tensor(np.eye(10)[y_val.reshape(-1)], dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=g_batch_size, shuffle=True)
    
    if model_flag == 0: # mlp
        print('MLP')
        model = SimpleMLP().to(device)
        optimizer = optim.Adam(model.parameters(), lr=g_lr, weight_decay=g_weight_decay)
        loss = nn.CrossEntropyLoss()
    else: # cnn
        print('CNN')
        model = SimpleCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=g_lr, weight_decay=g_weight_decay)
        loss = nn.CrossEntropyLoss()
        
    writer = SummaryWriter(f'runs/{run_name}')
    for epoch in range(g_num_epochs):
        model.train()
        epoch_start = time.perf_counter()
        epoch_loss = 0
        correct_num = 0
        step = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            correct_num += (torch.argmax(y_pred, dim=1) == torch.argmax(y_batch, dim=1)).sum().item()
            
            
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
            val_correct_num = (torch.argmax(y_val_pred, dim=1) == torch.argmax(y_val_device, dim=1)).sum().item()
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
    
    # Log predictions for the first 50 samples to TensorBoard.
    # Assume each row in y_pred is a predicted label (0-9); data has shape (n, c, h, w).
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test.to(device))
    vis_data = X_test[:50]
    vis_pred = torch.argmax(y_test_pred[:50], dim=1).cpu()
    for i in range(10):
        # mask is a boolean vector where vis_pred equals i.
        mask = (vis_pred == i)
        # Only log when there are images predicted as i.
        if mask.sum() > 0:
            # Log images predicted as i to TensorBoard.
            writer.add_images(f'num={i}', vis_data[mask])
    
    # Save CSV: first column is ImageId, second is predicted label.
    y_test_label = torch.argmax(y_test_pred, dim=1).cpu().numpy()
    sub = pd.DataFrame({'ImageId': np.arange(1, n_test + 1), 'Label': y_test_label})
    print(sub)
    sub.to_csv(f'{run_name}_submission.csv', index=False)


if __name__ == '__main__':
    main()


