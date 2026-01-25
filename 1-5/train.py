import time

import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, random_split

from cifar_cnn import ResNet
from dataset import CIFAR10TestDataset, CIFAR10TrainDataset
import utils

run_name = "v5"

g_eval_batch_size = 10000
g_num_epochs = 200
g_lr = 0.1
g_batch_size = 256
g_weight_decay = 5e-4

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


def  load_or_create_cache(cache_path, dataset):
    if os.path.exists(cache_path):
        print(f"[cache] load: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    print(f"[cache] create: {cache_path}")
    samples = [dataset[i] for i in range(len(dataset))]
    images = torch.stack([s[0] for s in samples])
    labels = torch.tensor([s[1] for s in samples])
    cache = {"images": images, "labels": labels}
    torch.save(cache, cache_path)
    return cache


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
    cache_dir = os.path.join(dataset_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    train_dataset = CIFAR10TrainDataset(
        images_dir='./dataset/train',
        labels_csv='./dataset/trainLabels.csv',
        class_to_idx=class_to_idx,
    )
    test_dataset = CIFAR10TestDataset(
        images_dir='./dataset/test',
    )

    if args.cache_data:
        train_cache = load_or_create_cache(
            os.path.join(cache_dir, 'train_cache.pt'),
            train_dataset,
        )
        test_cache = load_or_create_cache(
            os.path.join(cache_dir, 'test_cache.pt'),
            test_dataset,
        )
        train_dataset = TensorDataset(train_cache["images"], train_cache["labels"])
        test_dataset = TensorDataset(test_cache["images"], test_cache["labels"])
    
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
    # for data, target in train_loader:
    #     print(data.shape)
    #     print(target.shape)
    #     data_cpu = data.cpu()
    #     target_cpu = target.cpu()
    #     utils.draw_imgs(data_cpu, [idx_to_class[t.item()] for t in target_cpu])
    #     break
    
    # define model
    model = ResNet(3, 10).to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=g_lr,
        momentum=0.9,
        weight_decay=g_weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=g_num_epochs)
    loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_model_path = None
    best_val_acc = -1.0
    best_state_dict = None
    
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
        scheduler.step()
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(f"[best] epoch {epoch} val_acc={best_val_acc:.4f} cached in memory")
    
    
    # Test set is large; accumulate results in res_frame
    res_frame = pd.DataFrame(columns=['id', 'label'])
    
    print(f"test dataset size: {len(test_dataset)}")
    if best_state_dict is not None:
        best_model_path = os.path.join(
            "runs",
            run_name,
            f"best_model_{run_name}_acc_{best_val_acc:.4f}.pt",
        )
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        torch.save(best_state_dict, best_model_path)
        print(f"[best] save: {best_model_path}")
        model.load_state_dict(best_state_dict)
    model.eval()
    with torch.no_grad():
        for data, img_idx in test_loader:
            data = data.to(device, non_blocking=True)
            y_pred = model(data)
            y = torch.argmax(y_pred, dim=1).cpu()
            y = [idx_to_class[t.item()] for t in y]
            res_frame = pd.concat(
                [res_frame, pd.DataFrame({"id": img_idx, "label": y})],
                ignore_index=True,
            )
            print(f"{res_frame.shape[0]} / {len(test_dataset)}")

    res_frame = res_frame.sort_values("id")
    print(res_frame.shape)
    res_frame.to_csv("submission.csv", index=False)
