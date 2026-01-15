import os
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import utils

USE_CUDA = True
device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# 在这里定义你的模型，不要直接复制代码！
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()

        # 创建两个全连接层
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # 第一层全连接后使用ReLU激活函数
        x = self.fc1(x)
        x = F.relu(x)
        
        # 第二层全连接后直接输出
        return self.fc2(x)


class SimpleMLP3(nn.Module):
    def __init__(self):
        super(SimpleMLP3, self).__init__()

        # 创建两个全连接层
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        # 第1层全连接后使用ReLU激活函数
        x = self.fc1(x)
        x = F.relu(x)

        # 第2层全连接后使用ReLU激活函数
        x = self.fc2(x)
        x = F.relu(x)

        # 第3层全连接后直接输出
        x = self.fc3(x)
        x = F.relu(x)

        return self.fc4(x)

class SimpleMLP4(nn.Module):
    def __init__(self):
        super(SimpleMLP4, self).__init__()

        # 创建两个全连接层
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        # 第1层全连接后使用ReLU激活函数
        x = self.fc1(x)
        x = F.relu(x)

        # 第2层全连接后使用ReLU激活函数
        x = self.fc2(x)
        x = F.relu(x)

        # 第3层全连接后直接输出
        return self.fc3(x)

class SimpleMLP666(nn.Module):
    def __init__(self):
        super(SimpleMLP666, self).__init__()

        # 创建5个全连接层
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 1)

    def forward(self, x):
        # 第1层全连接后使用ReLU激活函数
        x = self.fc1(x)
        x = F.relu(x)

        # 第2层全连接后使用ReLU激活函数
        x = self.fc2(x)
        x = F.relu(x)
        
        # 第3层全连接后使用ReLU激活函数
        x = self.fc3(x)
        x = F.relu(x)
        
        # 第4层全连接后使用ReLU激活函数
        x = self.fc4(x)
        x = F.relu(x)

        # 第5层全连接后直接输出
        return self.fc5(x)

def fit_data3():
    # 在这里读入不同的 csv 文件
    X, Y = utils.read_csv_data(os.path.join(BASE_DIR, 'data_3.csv'))

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    print(X.shape, Y.shape)

    # 根据特征维度，绘制 2D 或 3D 散点图
    # utils.draw_2d_scatter(X, Y)

    # 模型、损失函数、优化器
    model = SimpleMLP3().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 创建数据集和加载器
    X = X.to(device)
    Y = Y.to(device)

    dataset = TensorDataset(X, Y)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 训练模型
    epochs = 200
    for epoch in range(epochs):
        epoch_loss = 0

        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            # 预测输出、计算损失
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            # 计算梯度、更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累积损失
            epoch_loss += loss.item()

        # 打印本轮的损失值
        print(f'Epoch {epoch}, Loss: {epoch_loss / len(data_loader)}')

    # 查看预测效果
    predicted = model(X)
    utils.draw_2d_scatter(
        X.detach().cpu().numpy(),
        Y.detach().cpu().numpy(),
        predicted.detach().cpu().numpy(),
    )
    # utils.draw_3d_scatter(X, Y, predicted.detach().numpy())

def fit_data4():
    # 在这里读入不同的 csv 文件
    X, Y = utils.read_csv_data(os.path.join(BASE_DIR, 'data_4.csv'))

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    print(X.shape, Y.shape)

    # 根据特征维度，绘制 2D 或 3D 散点图
    # utils.draw_2d_scatter(X, Y)
    utils.draw_3d_scatter(X, Y)

    # 模型、损失函数、优化器
    model = SimpleMLP4().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 创建数据集和加载器
    X = X.to(device)
    Y = Y.to(device)

    dataset = TensorDataset(X, Y)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 训练模型
    epochs = 200
    for epoch in range(epochs):
        epoch_loss = 0

        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            # 预测输出、计算损失
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            # 计算梯度、更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累积损失
            epoch_loss += loss.item()

        # 打印本轮的损失值
        print(f'Epoch {epoch}, Loss: {epoch_loss / len(data_loader)}')

    # 查看预测效果
    predicted = model(X)
    # utils.draw_2d_scatter(X, Y, predicted.detach().numpy())
    utils.draw_3d_scatter(
        X.detach().cpu().numpy(),
        Y.detach().cpu().numpy(),
        predicted.detach().cpu().numpy(),
    )


def fit_data666():
    # 在这里读入不同的 csv 文件
    X, Y = utils.read_csv_data(os.path.join(BASE_DIR, 'data_666.csv'))

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    print(X.shape, Y.shape)

    # 根据特征维度，绘制 2D 或 3D 散点图
    # utils.draw_2d_scatter(X, Y)

    # 模型、损失函数、优化器
    model = SimpleMLP666().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 创建数据集和加载器
    X = X.to(device)
    Y = Y.to(device)

    dataset = TensorDataset(X, Y)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 训练模型
    start_time = time.perf_counter()
    epochs = 200
    for epoch in range(epochs):
        epoch_loss = 0

        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            # 预测输出、计算损失
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            # 计算梯度、更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累积损失
            epoch_loss += loss.item()

        # 打印本轮的损失值
        print(f'Epoch {epoch}, Loss: {epoch_loss / len(data_loader)}')

    # 查看预测效果
    elapsed = time.perf_counter() - start_time
    print(f'Training time (fit_data666) [cuda={USE_CUDA}]: {elapsed:.4f}s')
    predicted = model(X)
    utils.draw_2d_scatter(
        X.detach().cpu().numpy(),
        Y.detach().cpu().numpy(),
        predicted.detach().cpu().numpy(),
    )
    # utils.draw_3d_scatter(X, Y, predicted.detach().numpy())


def demo():
    # 在这里读入不同的 csv 文件
    X, Y = utils.read_csv_data(os.path.join(BASE_DIR, 'data_4.csv'))

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    print(X.shape, Y.shape)

    # 根据特征维度，绘制 2D 或 3D 散点图
    # utils.draw_2d_scatter(X, Y)
    utils.draw_3d_scatter(X, Y)

    # 在这里开始你的表演，不要直接复制代码！
    # 模型、损失函数、优化器
    model = SimpleMLP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 创建数据集和加载器
    X = X.to(device)
    Y = Y.to(device)

    dataset = TensorDataset(X, Y)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 训练模型
    epochs = 100
    for epoch in range(epochs):
        epoch_loss = 0

        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            # 预测输出、计算损失
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            # 计算梯度、更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累积损失
            epoch_loss += loss.item()

        # 打印本轮的损失值
        print(f'Epoch {epoch}, Loss: {epoch_loss / len(data_loader)}')

    # 查看预测效果
    predicted = model(X)
    # utils.draw_2d_scatter(X, Y, predicted.detach().numpy())
    utils.draw_3d_scatter(
        X.detach().cpu().numpy(),
        Y.detach().cpu().numpy(),
        predicted.detach().cpu().numpy(),
    )

if __name__ == '__main__':
    # fit_data3()
    # fit_data4()
    fit_data666()
