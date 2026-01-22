import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, random_split

from cifar_cnn import ResNet
from dataset import CIFAR10TestDataset, CIFAR10TrainDataset
import utils

run_name = "00"

batch_size = 32
eval_batch_size = 10000

# dataset 的类别
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

# 类别到索引的映射
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(cifar10_classes)}
# 索引到类别的映射
idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}


if __name__ == '__main__':
    dataset_dir = './dataset'
    
    train_dataset = CIFAR10TrainDataset(
        images_dir='./dataset/train',
        labels_csv='./dataset/trainLabels.csv',
        class_to_idx=class_to_idx,
    )
    test_dataset = CIFAR10TestDataset(
        images_dir='./dataset/test',
    )
    
    train_size = int(len(train_dataset) * 0.8)
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
    )
    
    print(f"训练集样本数: {len(train_dataset)}")
    for data, target in train_loader:
        print(data.shape)
        print(target.shape)
        utils.draw_imgs(data, [idx_to_class[t.item()] for t in target])
        break
    
    
    # 因测试集过大，无法一次推理出来，因此用 res_frame 保存结果
    res_frame = pd.DataFrame(columns=['id', 'label'])
    
    print(f"测试集样本数: {len(test_dataset)}")
    for data, img_idx in test_loader:
        print(data.shape)
        print(img_idx.shape)
        
        # y 为模型预测的标签索引，这里先随机生成
        y = torch.randint(0, 10, (data.size(0),))
        
        # 将 y 中索引转换为类别名
        y = [idx_to_class[t.item()] for t in y]

        # 将预测结果添加到 res_frame 中
        res_frame = pd.concat([res_frame, pd.DataFrame({'id': img_idx, 'label': y})], ignore_index=True)

        # 打印进度
        print(f'{res_frame.shape[0]} / {len(test_dataset)}')

    # 按图片索引排序
    res_frame = res_frame.sort_values('id')
    print(res_frame.shape)

    # 保存结果
    res_frame.to_csv('submission.csv', index=False)
