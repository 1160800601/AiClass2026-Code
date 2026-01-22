import time

import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, random_split

from cifar_cnn import ResNet
from dataset import CIFAR10TestDataset, CIFAR10TrainDataset
import utils

run_name = "01"

g_eval_batch_size = 10000
g_num_epochs = 200
g_lr = 0.01
g_batch_size = 512
g_weight_decay = 1e-4

# Dataset classes
cifar10_classes = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

# Class-to-index mapping
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(cifar10_classes)}
# Index-to-class mapping
idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}


class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.samples = [dataset[i] for i in range(len(dataset))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-data", action="store_true", default=True)
    parser.add_argument("--no-cache-data", action="store_false", dest="cache_data")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Select device.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    
    dataset_dir = './dataset'
    
    train_dataset = CIFAR10TrainDataset(
        images_dir='./dataset/train',
        labels_csv='./dataset/trainLabels.csv',
        class_to_idx=class_to_idx,
    )
    test_dataset = CIFAR10TestDataset(
        images_dir='./dataset/test',
    )

    if args.cache_data:
        train_dataset = CachedDataset(train_dataset)
        test_dataset = CachedDataset(test_dataset)
    
    train_size = int(len(train_dataset) * 0.8)
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    n_train = len(train_subset)

    train_loader = DataLoader(
        train_subset,
        batch_size=g_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=g_eval_batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=g_eval_batch_size,
        shuffle=False,
    )
    
    print(f"train subset size: {len(train_subset)}")
    print(f"val subset size: {len(val_subset)}")
    for data, target in train_loader:
        print(data.shape)
        print(target.shape)
        data_cpu = data.cpu()
        target_cpu = target.cpu()
        utils.draw_imgs(data_cpu, [idx_to_class[t.item()] for t in target_cpu])
        break
    
    # define model
    model = ResNet(3, 10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=g_lr, weight_decay=g_weight_decay)
    loss = nn.CrossEntropyLoss()
    
    writer = SummaryWriter(f'runs/{run_name}')
    
    # training model
    for epoch in range(g_num_epochs):
        model.train()
        epoch_start = time.perf_counter()
        epoch_loss = 0
        correct_num = 0
        step = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            correct_num += (torch.argmax(y_pred, dim=1) == y_batch).sum().item()
            
            
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
            X_val, y_val = next(iter(val_loader))
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            y_val_pred = model(X_val)
            val_correct_num = (torch.argmax(y_val_pred, dim=1) == y_val).sum().item()
            val_accuracy = val_correct_num / g_eval_batch_size
        
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
    
    
    # Test set is large; accumulate results in res_frame
    res_frame = pd.DataFrame(columns=['id', 'label'])
    
    print(f"test dataset size: {len(test_dataset)}")
    for data, img_idx in test_loader:
        print(data.shape)
        print(img_idx.shape)
        # y is the predicted label index; generate random values here
        y = torch.randint(0, 10, (data.size(0),))
        # Convert y indices to class names
        y = [idx_to_class[t.item()] for t in y]
        # Append predictions to res_frame
        res_frame = pd.concat([res_frame, pd.DataFrame({'id': img_idx, 'label': y})], ignore_index=True)
        # Print progress
        print(f'{res_frame.shape[0]} / {len(test_dataset)}')

    # Sort by image index
    res_frame = res_frame.sort_values('id')
    print(res_frame.shape)

    # Save results
    res_frame.to_csv('submission.csv', index=False)
