01:初步模型调通
02：移动至cuda
    cuda:60s   106 epoch
    cpu: 108s  200 epoch
03：统计数据
    50轮后应该是有点过拟合了
    val acc: 0.765
04: 训练加入dropout，调整batchsize
    val acc: 0.79
    40轮后过拟合, val acc下降