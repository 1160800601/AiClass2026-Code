import utils
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd
from mlp import SimpleMLP
from cnn import SimpleCNN

# 自己构建的数据集路径
img_dir = './handwrite'
# run_name = 'mlp02'
# model_flag = 0  # 0: mlp, 1: cnn
run_name = 'cnn06'
model_flag = 1  # 0: mlp, 1: cnn

eval_ver = 'v1'

def main():
    # Select device.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    # 把指定路径下的图片全部读入，转为 28x28 的灰度图，返回 ndarray
    img_data = utils.read_img_from_dir(img_dir, img_size=(28, 28), gray=True)
    print(img_data.shape)
    print(img_data.dtype)
    print(img_data[0])
    utils.draw_imgs(img_data)
    
    # 归一化
    img_data = img_data / 255.0 
    
    # 反转灰度（如果图片是白底黑字，使用反转灰度）
    img_data = 1 - img_data
    img_data = np.where(img_data > 0.1, 1.0, img_data)
    utils.draw_imgs(img_data)
    
    # 加载模型
    if model_flag == 0:
        model = SimpleMLP().to(device)
        model.load_state_dict(torch.load(f'{run_name}.pt', map_location=device, weights_only=True))
    else:
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load(f'{run_name}.pt', map_location=device, weights_only=True))
    
    # 预测
    model.eval()
    data = torch.tensor(img_data, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(data)
        y_pred = torch.argmax(logits, dim=1)
    print(y_pred.cpu().numpy())
    
    # 用 tensorboard 记录预测结果
    # 假设 y_pred 中每一行是预测的标签（数字0～9）；data 是对应的图片，形状是 (n, c, h, w)
    writer = SummaryWriter(f'runs/eval_{run_name}_{eval_ver}')
    for i in range(10):
        # mask 是一个布尔向量，表示 y_pred 的值等于 i 的位置，即预测为数字 i 的位置
        mask = (y_pred.view(-1) == i)
        # 仅当存在预测为数字 i 的图片时才记录
        if mask.sum() > 0:
            # 把预测为数字 i 的图片记录到 tensorboard
            writer.add_images(f'num={i}', data[mask].detach().cpu())
    writer.close()


if __name__ == '__main__':
    main()
