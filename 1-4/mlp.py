import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义一个简单的多层感知机模型
# pytorch 中所有自定义的神经网络类，都要继承 nn.Module
class SimpleMLP(nn.Module):
    # TODO: 开始你的表演，请不要直接复制代码！