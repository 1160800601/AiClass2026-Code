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
- residuals per stage: 3
- stem: Conv2d k3 s1 p1 -> BN -> ReLU
- block: two 3x3 convs (p1); first block stride 1 (stage 0) else 2; 1x1 skip when needed
- head: AdaptiveAvgPool2d(1x1) -> Flatten -> Linear(512, num_classes)
- score:

