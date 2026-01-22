from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
    """实现残差块"""
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        # 是否使用 1x1 卷积层来适配尺寸
        if use_1x1conv:
            self.res_conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, stride=stride
            )
        else:
            self.res_conv = None

    def forward(self, X):
        Y = self.seq(X)

        if self.res_conv:
            X = self.res_conv(X)

        Y += X
        return F.relu(Y)


# 适配 cifar-10 数据集的卷积神经网络
class ResNet(nn.Module):
    
    def __init__(self, input_chnls, num_classes):
        super().__init__()
        
        # 1. stage chnls
        self.chnl_cfg = [64, 128, 256, 512]
        self.num_residuals = 2
        
        # 2. input
        modules = [
            nn.Conv2d(input_chnls, self.chnl_cfg[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]
        
        # 3. loop residual
        in_chnl_num = self.chnl_cfg[0]
        for i, out_chnl_num in enumerate(self.chnl_cfg):
            is_first_stage = (i == 0)
            modules.append(
                self._make_layer(in_chnl_num, out_chnl_num, self.num_residuals, fisrt_block=is_first_stage)
            )
            in_chnl_num = out_chnl_num

        # 4. output
        modules.extend([nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(self.chnl_cfg[-1], num_classes)])
    
    def _make_layer(self, in_chnl_num, out_chnl_num, num_residuals, first_block = False):
        layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                layers.append(Residual(in_chnl_num, out_chnl_num, use_1x1conv=True, stride=2))
            elif i == 0 and first_block:
                layers.append(Residual(in_chnl_num, out_chnl_num, use_1x1conv=(in_chnl_num != out_chnl_num)))
            else:
                layers.append(Residual(in_chnl_num, out_chnl_num))
            return nn.Sequential(*layers)
            
        
    def forward(self, x):
        x = self.net(x)
        return x
