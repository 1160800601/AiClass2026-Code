from torch import nn
from embedding.positional_encoding import PositionalEncoding


class TransformerEmbedding(nn.Module):
    """Transformer 的词嵌入层，包括 TokenEmbedding 和 PositionalEncoding
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, pad_idx, device):
        """
        :param vocab_size: 词汇表大小
        :param d_model: 模型的维度
        :param max_len: 最大序列长度
        :param drop_prob: 丢弃概率（用于 Dropout）
        :param pad_idx: 填充token的索引
        :param device: 硬件设备设置
        """
        super(TransformerEmbedding, self).__init__()

        # 词嵌入层 (Token Embedding)
        # nn.Embedding 本质是一个可学习的查找表，将离散的 token ID 映射为连续的向量表示
        # - num_embeddings: 查找表的行数，即词汇表大小
        # - embedding_dim: 每个嵌入向量的维度，即 d_model
        # - padding_idx: 填充 token 的索引，该位置的嵌入向量固定为全 0 且不参与梯度更新
        # 输入形状: [batch_size, seq_len] (token ID)
        # 输出形状: [batch_size, seq_len, d_model] (嵌入向量)
        self.tok_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=pad_idx)
        
        # 位置编码
        self.pos_emb = PositionalEncoding(d_model, max_len, device)

        # Dropout 层
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """
        :param x: 输入的 token 索引，形状为 [batch_size, seq_len]
        :return: 加上位置编码和丢弃层后的嵌入表示
        """
        # x.shape: [batch_size, seq_len]
        # 计算词嵌入
        tok_emb = self.tok_emb(x)
        # tok_emb.shape: [batch_size, seq_len, d_model]

        # 计算位置编码
        pos_emb = self.pos_emb(x)
        # pos_emb.shape: [seq_len, d_model]

        # 返回加了位置编码和 Dropout 的词嵌入
        return self.drop_out(tok_emb + pos_emb) # 通过广播机制，将pos_emb扩展为与tok_emb相同的形状
