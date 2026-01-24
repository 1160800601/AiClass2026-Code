## base line
- model: ResNet (CIFAR-10)
- stage channels: [64, 128, 256, 512]
- residuals per stage: 2
- stem: Conv2d k7 s2 p3 -> BN -> ReLU -> MaxPool k3 s2 p1
- block: two 3x3 convs (p1); first block stride 1 (stage 0) else 2; 1x1 skip when needed
- head: AdaptiveAvgPool2d(1x1) -> Flatten -> Linear(512, num_classes)
- score:0.658

