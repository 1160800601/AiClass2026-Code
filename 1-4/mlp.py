import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义一个简单的多层感知机模型
# pytorch 中所有自定义的神经网络类，都要继承 nn.Module
class SimpleMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        input_dim = 28 * 28
        output_dim = 10

        layers = []
        dim = input_dim
        for hidden_dim in (256, 128):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        
        return self.net(x)
    
