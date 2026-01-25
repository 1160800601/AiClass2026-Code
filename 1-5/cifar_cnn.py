from torch import nn
from torch.nn import functional as F

RESIDUAL_ENABLED = True

class Residual(nn.Module):
    """Residual block."""
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1, num_convs=2):
        super().__init__()
        if num_convs < 1:
            raise ValueError("num_convs must be >= 1")
        layers = []
        for i in range(num_convs):
            in_chnl = in_channels if i == 0 else out_channels
            conv_stride = stride if i == 0 else 1
            layers.extend([
                nn.Conv2d(in_chnl, out_channels, kernel_size=3, padding=1, stride=conv_stride),
                nn.BatchNorm2d(out_channels),
            ])
            if i != num_convs - 1:
                layers.append(nn.ReLU())
        self.seq = nn.Sequential(*layers)

        # Use 1x1 conv to match shapes when needed.
        if use_1x1conv and RESIDUAL_ENABLED:
            self.res_conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, stride=stride
            )
        else:
            self.res_conv = None

    def forward(self, X):
        Y = self.seq(X)

        if RESIDUAL_ENABLED and self.res_conv:
            X = self.res_conv(X)

        if RESIDUAL_ENABLED:
            Y += X
        return F.relu(Y)


# CNN adapted for CIFAR-10.
class ResNet(nn.Module):
    
    def __init__(self, input_chnls, num_classes):
        super().__init__()
        
        # 1. stage chnls
        # self.chnl_cfg = [64, 128, 256, 512]
        self.chnl_cfg = [32, 64, 128, 256]
        self.res_cfg = [3, 4, 6, 3]
        
        # 2. input
        modules = [
            # nn.Conv2d(input_chnls, self.chnl_cfg[0], kernel_size=7, stride=2, padding=3),
            nn.Conv2d(input_chnls, self.chnl_cfg[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.chnl_cfg[0]),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]
        
        # 3. loop residual
        in_chnl_num = self.chnl_cfg[0]
        for i, out_chnl_num in enumerate(self.chnl_cfg):
            is_first_stage = (i == 0)
            modules.append(
                self._make_layer(in_chnl_num, out_chnl_num, self.res_cfg[i], first_stage=is_first_stage)
            )
            in_chnl_num = out_chnl_num
            
        modules.append(nn.Dropout(p=0.2))
        
        # 4. output
        modules.extend([nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(self.chnl_cfg[-1], num_classes)])
        
        self.net = nn.Sequential(*modules)
    
    def _make_layer(self, in_chnl_num, out_chnl_num, num_residuals, first_stage = False):
        layers = []
        for i in range(num_residuals):
            if i == 0:
                stride = 1 if first_stage else 2
                use_1x1conv = (not first_stage) or (in_chnl_num != out_chnl_num)
                layers.append(
                    Residual(in_chnl_num, out_chnl_num, use_1x1conv=use_1x1conv, stride=stride)
                )
            else:
                layers.append(Residual(out_chnl_num, out_chnl_num))
        return nn.Sequential(*layers)
            
        
    def forward(self, x):
        x = self.net(x)
        return x

