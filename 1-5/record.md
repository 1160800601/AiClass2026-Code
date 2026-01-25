## base line
- model: ResNet (CIFAR-10)
- stage channels: [64, 128, 256, 512]
- residuals per stage: 2
- stem: Conv2d k7 s2 p3 -> BN -> ReLU -> MaxPool k3 s2 p1
- block: two 3x3 convs (p1); first block stride 1 (stage 0) else 2; 1x1 skip when needed
- head: AdaptiveAvgPool2d(1x1) -> Flatten -> Linear(512, num_classes)
- score:0.658

## v1
- change: stem 结构修改
- model: ResNet (CIFAR-10)
- stage channels: [64, 128, 256, 512]
- residuals per stage: 2
- stem: Conv2d k3 s1 p1 -> BN -> ReLU
- block: two 3x3 convs (p1); first block stride 1 (stage 0) else 2; 1x1 skip when needed
- head: AdaptiveAvgPool2d(1x1) -> Flatten -> Linear(512, num_classes)
- score:0.763

## v2
- change: residual加深
- model: ResNet (CIFAR-10)
- stage channels: [64, 128, 256, 512]
- residuals per stage: 4
- stem: Conv2d k3 s1 p1 -> BN -> ReLU
- block: two 3x3 convs (p1); first block stride 1 (stage 0) else 2; 1x1 skip when needed
- head: AdaptiveAvgPool2d(1x1) -> Flatten -> Linear(512, num_classes)
- score:0.752

## v3
- change: stage channels 翻倍, 学习率增加至0.1
- model: ResNet (CIFAR-10)
- stage channels: [128, 256, 512， 1024]
- residuals per stage: 2
- stem: Conv2d k3 s1 p1 -> BN -> ReLU
- block: two 3x3 convs (p1); first block stride 1 (stage 0) else 2; 1x1 skip when needed
- head: AdaptiveAvgPool2d(1x1) -> Flatten -> Linear(512, num_classes)
- score: 训不出来


## v4
- change: stage channels 翻倍, 学习率增加至0.1
- model: ResNet (CIFAR-10)
- stage channels: [64, 128, 256, 512]
- residuals per stage: 2
- stem: Conv2d k3 s1 p1 -> BN -> ReLU
- g_eval_batch_size = 10000
- g_num_epochs = 50
- g_lr = 0.1
- g_batch_size = 512
- g_weight_decay = 1e-4
- block: two 3x3 convs (p1); first block stride 1 (stage 0) else 2; 1x1 skip when needed
- head: AdaptiveAvgPool2d(1x1) -> Flatten -> Linear(512, num_classes)
- score: 0.5  很差
  

## v5
- change: stem
- model: ResNet (CIFAR-10)
- stage channels: [32, 64, 128, 256]
- residuals per stage: 2
- stem: Conv2d k3 s1 p1 -> BN -> ReLU
- g_eval_batch_size = 10000
- g_num_epochs = 50
- g_lr = 0.01
- g_batch_size = 256
- g_weight_decay = 1e-4
- block: two 3x3 convs (p1); first block stride 1 (stage 0) else 2; 1x1 skip when needed
- head: AdaptiveAvgPool2d(1x1) -> Flatten -> Linear(256, num_classes)
- score: 0.781

## v6
- change: optimizer
- model: ResNet (CIFAR-10)
- stage channels: [32, 64, 128, 256]
- residuals per stage: [3, 4, 6, 3]
- stem: Conv2d k3 s1 p1 -> BN -> ReLU
- g_eval_batch_size = 10000
- g_num_epochs = 200
- g_lr = 0.1
- g_batch_size = 256
- g_weight_decay = 5e-4
- optimizer: SGD (momentum=0.9, nesterov=True)
- scheduler: cosine (CosineAnnealingLR)
- block: two 3x3 convs (p1); first block stride 1 (stage 0) else 2; 1x1 skip when needed
- head: AdaptiveAvgPool2d(1x1) -> Flatten -> Linear(256, num_classes)
- 问题：Epoch: 91, Train Loss: 0.1012, Train Acc: 0.9645, Val Acc: 0.7140, Time: 6.13s，过拟合了
- 
## v7
- change: label_smoothing、dropout， 数据增强
- model: ResNet (CIFAR-10)
- stage channels: [32, 64, 128, 256]
- residuals per stage: [3, 4, 6, 3]
- stem: Conv2d k3 s1 p1 -> BN -> ReLU
- g_eval_batch_size = 10000
- g_num_epochs = 200
- g_lr = 0.1
- g_batch_size = 256
- g_weight_decay = 5e-4
- optimizer: SGD (momentum=0.9, nesterov=True)
- scheduler: cosine (CosineAnnealingLR)
- block: two 3x3 convs (p1); first block stride 1 (stage 0) else 2; 1x1 skip when needed
- head: AdaptiveAvgPool2d(1x1) -> Flatten -> Linear(256, num_classes)