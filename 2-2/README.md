# 从零实现 Transformer：英译中翻译任务

本项目从零开始实现了 Transformer 模型，用于英语到中文的翻译任务。

## 📖 目录

- [Transformer 架构概述](#transformer-架构概述)
- [项目结构](#项目结构)
- [核心组件详解](#核心组件详解)
  - [1. 缩放点积注意力](#1-缩放点积注意力-scale-dot-product-attention)
  - [2. 多头注意力机制](#2-多头注意力机制-multi-head-attention)
  - [3. 前馈神经网络](#3-前馈神经网络-feed-forward-network)
  - [4. 位置编码](#4-位置编码-positional-encoding)
  - [5. Transformer Embedding](#5-transformer-embedding)
  - [6. 编码器 Block](#6-编码器-block)
  - [7. 解码器 Block](#7-解码器-block)
  - [8. 完整 Transformer](#8-完整-transformer)
- [快速开始](#快速开始)
- [配置参数](#配置参数)

---

## Transformer 架构概述

Transformer 是 2017 年 Google 在论文 [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) 中提出的革命性架构。它完全基于注意力机制，抛弃了传统的循环神经网络（RNN），在机器翻译任务中取得了显著成果。

### 架构图示

```
┌─────────────────┐
                    │     Output      │
                    │   Probabilities │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │     Linear      │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼─────────┐         ┌─────────▼─────────┐
    │                   │         │                   │
    │     Encoder       │────────▶│     Decoder       │
    │   (N blocks)      │  enc    │   (N blocks)      │
    │                   │  output │                   │
    └─────────▲─────────┘         └─────────▲─────────┘
              │                             │
    ┌─────────┴─────────┐         ┌─────────┴─────────┐
    │    Embedding +    │         │    Embedding +    │
    │ Positional Encode │         │ Positional Encode │
    └─────────▲─────────┘         └─────────▲─────────┘
              │                             │
      ┌───────┴───────┐             ┌───────┴───────┐
      │ Source Input  │             │ Target Input  │
      │   (English)   │             │   (Chinese)   │
      └───────────────┘             └───────────────┘
```

---

## 项目结构

```
2-2/
├── layers/                          # 基础层
│   ├── scale_dot_product_attention.py  # 缩放点积注意力
│   ├── multi_head_attention.py         # 多头注意力机制
│   └── ffn.py                          # 前馈神经网络
│
├── embedding/                       # 嵌入层
│   ├── positional_encoding.py          # 正弦位置编码
│   └── transformer_embedding.py        # Token嵌入 + 位置编码
│
├── blocks/                          # Transformer 块
│   ├── encoder_block.py                # 编码器 Block
│   ├── encoder.py                      # 完整编码器
│   ├── decoder_block.py                # 解码器 Block
│   └── decoder.py                      # 完整解码器
│
├── transformer.py                   # Transformer 完整模型
├── config.py                        # 配置参数
├── text_process.py                  # 文本预处理
├── train.py                         # 训练脚本
├── eval.py                          # 推理脚本
└── vocab.json                       # 词表文件
```

---

## 核心组件详解

### 1. 缩放点积注意力 (Scale Dot-Product Attention)

> 📁 文件：`layers/scale_dot_product_attention.py`

这是 Transformer 中最核心的计算单元。注意力机制的本质是：**根据查询（Query）和键（Key）的相似度，对值（Value）进行加权求和。**

**数学公式：**

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中 $d_k$ 是键向量的维度，除以 $\sqrt{d_k}$ 是为了防止点积结果过大导致 softmax 梯度消失。

**代码实现：**

```python
def forward(self, Q, K, V, mask=None):
    # Q, K, V 形状: [batch_size, head, length, d_tensor]
  
    # 1. 计算注意力分数矩阵
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_tensor))
  
    # 2. 应用掩码（用于解码器的因果遮蔽）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -10000)
  
    # 3. Softmax 归一化
    attn_weights = torch.softmax(scores, dim=-1)
  
    # 4. 加权求和
    attn_output = torch.matmul(attn_weights, V)
  
    return attn_output
```

**形状变化：**

```
Q × K^T: [B, H, L, D] × [B, H, D, L] → [B, H, L, L]  (注意力权重矩阵)
attn × V: [B, H, L, L] × [B, H, L, D] → [B, H, L, D]  (加权后的输出)
```

---

### 2. 多头注意力机制 (Multi-Head Attention)

> 📁 文件：`layers/multi_head_attention.py`

多头注意力允许模型在不同的表示子空间中并行学习信息，就像同时从多个角度"关注"输入序列。

**核心思想：**

- 将 Q、K、V 分别投影到多个不同的子空间
- 在每个子空间独立计算注意力
- 将所有头的输出拼接后，再进行一次线性变换

**代码实现：**

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8):
        self.head_dim = hidden_dim // num_heads  # 每个头的维度
  
        # 四个投影矩阵
        self.W_q = nn.Linear(hidden_dim, hidden_dim)  # Query 投影
        self.W_k = nn.Linear(hidden_dim, hidden_dim)  # Key 投影
        self.W_v = nn.Linear(hidden_dim, hidden_dim)  # Value 投影
        self.W_o = nn.Linear(hidden_dim, hidden_dim)  # 输出投影
  
    def forward(self, x_q, x_k, x_v, mask=None):
        # 1. 线性投影
        Q = self.W_q(x_q)
        K = self.W_k(x_k)
        V = self.W_v(x_v)
  
        # 2. 分割多头 [B, L, D] → [B, H, L, D/H]
        Q = self._split(Q)
        K = self._split(K)
        V = self._split(V)
  
        # 3. 计算注意力
        attn_output = self.attention(Q, K, V, mask)
  
        # 4. 拼接多头并输出投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, hidden_dim)
        output = self.W_o(attn_output)
  
        return output
```

---

### 3. 前馈神经网络 (Feed-Forward Network)

> 📁 文件：`layers/ffn.py`

FFN 是一个简单的两层全连接网络，用于对每个位置的表示进行非线性变换。

$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

```python
class FFN(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        self.linear1 = nn.Linear(d_model, hidden)   # 512 → 1024
        self.linear2 = nn.Linear(hidden, d_model)   # 1024 → 512
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
```

---

### 4. 位置编码 (Positional Encoding)

> 📁 文件：`embedding/positional_encoding.py`

由于 Transformer 没有循环结构，无法感知序列中 token 的位置信息。**位置编码**通过给每个位置添加一个固定的向量来解决这个问题。

**正弦位置编码公式：**

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

**代码实现：**

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super().__init__()
  
        # 创建位置编码矩阵
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # 不参与训练
  
        # 位置索引 [0, 1, 2, ..., max_len-1]
        pos = torch.arange(0, max_len, device=device).float().unsqueeze(1)
  
        # 维度索引
        k = torch.arange(0, d_model / 2, device=device).float()
  
        # 计算正弦和余弦
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (2 * k / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (2 * k / d_model)))
```

**为什么用正弦/余弦？**

- 可以表示任意长度的序列
- 相对位置可以通过线性变换得到
- 值域固定在 [-1, 1]

---

### 5. Transformer Embedding

> 📁 文件：`embedding/transformer_embedding.py`

将 Token Embedding 和 Positional Encoding 结合：

```python
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, pad_idx, device):
        # Token 嵌入层
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        # 位置编码
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        # Dropout
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)           # [B, L] → [B, L, D]
        pos_emb = self.pos_emb(x)           # [L, D]
        return self.drop_out(tok_emb + pos_emb)  # 广播相加
```

---

### 6. 编码器 Block

> 📁 文件：`blocks/encoder_block.py`

每个编码器 Block 包含两个子层：

1. **多头自注意力层** (Multi-Head Self-Attention)
2. **前馈神经网络层** (Feed-Forward Network)

每个子层都有 **残差连接** 和 **层归一化**。

```
         ┌───────────────────┐
         │      输入 x       │
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │  Multi-Head Attn  │◄── Self-Attention
         └─────────┬─────────┘
                   │
    ┌──────────────┼──────────────┐
    │              ▼              │
    │    Dropout + LayerNorm      │◄── 残差连接
    │              │              │
    └──────────────┼──────────────┘
                   │
         ┌─────────▼─────────┐
         │       FFN         │
         └─────────┬─────────┘
                   │
    ┌──────────────┼──────────────┐
    │              ▼              │
    │    Dropout + LayerNorm      │◄── 残差连接
    │              │              │
    └──────────────┼──────────────┘
                   │
         ┌─────────▼─────────┐
         │      输出         │
         └───────────────────┘
```

**代码实现：**

```python
def forward(self, x):
    # 1. 自注意力
    x_origin = x
    x = self.attention(x, x, x)  # Q=K=V=x (自注意力)
  
    # 2. 残差连接 + 层归一化
    x = self.dropout1(x)
    x = self.norm1(x + x_origin)
  
    # 3. 前馈网络
    x_origin = x
    x = self.ffn(x)
  
    # 4. 残差连接 + 层归一化
    x = self.dropout2(x)
    x = self.norm2(x + x_origin)
  
    return x
```

---

### 7. 解码器 Block

> 📁 文件：`blocks/decoder_block.py`

解码器 Block 比编码器多了一个 **Cross-Attention** 层，用于关注编码器的输出：

```
┌───────────────────┐
         │    目标输入 x     │
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │  Masked Self-Attn │◄── 带掩码的自注意力
         └─────────┬─────────┘
                   │
    ┌──────────────┼──────────────┐
    │    Dropout + LayerNorm     │
    └──────────────┼──────────────┘
                   │
         ┌─────────▼─────────┐
         │   Cross-Attention │◄── Q来自解码器，K/V来自编码器
    ┌────│                   │────┐
    │    └─────────┬─────────┘    │
    │              │         Encoder Output
    └──────────────┼──────────────┘
                   │
    ┌──────────────┼──────────────┐
    │    Dropout + LayerNorm     │
    └──────────────┼──────────────┘
                   │
         ┌─────────▼─────────┐
         │       FFN         │
         └─────────┬─────────┘
                   │
    ┌──────────────┼──────────────┐
    │    Dropout + LayerNorm     │
    └──────────────┼──────────────┘
                   │
         ┌─────────▼─────────┐
         │      输出         │
         └───────────────────┘
```

**关键区别：**

- **Masked Self-Attention**: 使用下三角掩码，防止模型"偷看"未来的 token
- **Cross-Attention**: Query 来自解码器，Key 和 Value 来自编码器输出

```python
def forward(self, x, enc, trg_mask):
    # 1. 带掩码的自注意力
    x_origin = x
    x = self.self_attention(x, x, x, mask=trg_mask)  # 添加掩码
    x = self.dropout1(x)
    x = self.norm1(x + x_origin)
  
    # 2. Cross-Attention（编码器-解码器注意力）
    x_origin = x
    x = self.enc_dec_attention(x_q=x, x_k=enc, x_v=enc)  # Q来自x，K/V来自enc
    x = self.dropout2(x)
    x = self.norm2(x + x_origin)
  
    # 3. 前馈网络
    x_origin = x
    x = self.ffn(x)
    x = self.dropout3(x)
    x = self.norm3(x + x_origin)
  
    return x
```

---

### 8. 完整 Transformer

> 📁 文件：`transformer.py`

将所有组件组合成完整的 Transformer 模型：

```python
class Transformer(nn.Module):
    def __init__(self, ...):
        # 编码器
        self.encoder = Encoder(...)
        # 解码器
        self.decoder = Decoder(...)
  
    def forward(self, src, trg):
        # 1. 生成目标序列的下三角掩码
        mask = self.make_mask(trg)
  
        # 2. 编码源序列
        enc_src = self.encoder(src)
  
        # 3. 解码目标序列
        output = self.decoder(trg, enc_src, mask)
  
        return output
  
    def make_mask(self, trg):
        """生成下三角掩码，防止模型看到未来的 token"""
        trg_len = trg.shape[1]
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)).bool()
        return trg_mask.unsqueeze(0).repeat(trg.size(0), 1, 1)
```

---

## 快速开始

### 环境要求

```bash
pip install torch nltk
```

### 训练模型

```bash
cd 2-2
python train.py
```

训练过程会：

1. 构建词表并保存到 `vocab.json`
2. 使用内置的英译中数据集进行训练
3. 将模型保存到 `transformer.pth`

### 推理测试

```bash
python eval.py
```

然后输入英文句子，模型会输出对应的中文翻译：

```
Using device: mps
Input english: Good morning!
Output chinese: <bos>早上好！<eos>
```

---

## 配置参数

> 📁 文件：`config.py`


| 参数         | 默认值 | 说明                |
| ------------ | ------ | ------------------- |
| `d_model`    | 512    | 模型/嵌入维度       |
| `n_head`     | 8      | 多头注意力的头数    |
| `max_len`    | 40     | 最大序列长度        |
| `ffn_hidden` | 1024   | FFN 隐藏层维度      |
| `n_blocks`   | 2      | 编码器/解码器的层数 |
| `drop_prob`  | 0.1    | Dropout 概率        |
| `batch_size` | 20     | 批次大小            |
| `lr`         | 0.001  | 学习率              |
| `epochs`     | 200    | 训练轮数            |

---

## 文本处理流程

> 📁 文件：`text_process.py`

```
输入: "Hello World!"
    ↓ tokenize (分词)
['Hello', 'World', '!']
    ↓ add_special_token (添加特殊标记)
['<bos>', 'Hello', 'World', '!', '<eos>', '<pad>', '<pad>', ...]
    ↓ vocab lookup (词表映射)
[0, 45, 78, 12, 1, 2, 2, ...]
    ↓ to tensor
tensor([0, 45, 78, 12, 1, 2, 2, ...])
```

**特殊 Token：**

- `<bos>`: 序列开始标记 (Beginning of Sequence)
- `<eos>`: 序列结束标记 (End of Sequence)
- `<pad>`: 填充标记 (Padding)
- `<unk>`: 未知词标记 (Unknown)

---

## 训练技巧

### Teacher Forcing

训练时使用 **Teacher Forcing** 策略：解码器的输入是真实的目标序列（去掉最后一个 token），而不是模型自己的预测结果。

```python
# target[:, :-1] 作为解码器输入（去掉 <eos>）
output = model(input, target[:, :-1])

# target[:, 1:] 作为训练标签（去掉 <bos>）
loss = criterion(output.reshape(-1, vocab_size), target[:, 1:].reshape(-1))
```

### 自回归生成

推理时使用 **自回归生成**：每次生成一个 token，并将其加入到解码器输入中。

```python
target = torch.tensor([[vocab['<bos>']]])  # 初始化为 <bos>

for _ in range(max_len - 1):
    output = model(input_data, target)
    next_token = output[:, -1, :].argmax(dim=-1)  # 取最后一个位置的预测
    target = torch.cat([target, next_token.unsqueeze(1)], dim=1)
  
    if next_token.item() == vocab['<eos>']:
        break
```

---

Happy Learning! 🚀
