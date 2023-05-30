import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from embed import DataEmbedding

class TransformerModel(nn.Module):

    def __init__(self, n_feature: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        """
               初始化TimeVAE模型。

               参数:
               n_feature (int): 特征数量。
               d_model (int): 模型的隐藏层维度。
               nhead (int): 多头注意力机制的头数。
               d_hid (int): 前馈网络的隐藏层维度。
               nlayers (int): Transformer编码器层的数量。
               dropout (float): Dropout比率，默认为0.5。
               """
        super().__init__()
        self.model_type = 'Transformer'
        # 创建Transformer编码器层
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        # 创建Transformer编码器层
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # 创建Transformer编码器层
        self.encoder = DataEmbedding(n_feature, d_model, dropout=0.0)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, n_feature)


    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """
       生成一个上三角矩阵，所有元素为 ``-inf``，对角线及其下方的元素为 ``0``。

       参数:
       sz (int): 矩阵的大小。

       返回:
       torch.Tensor: 形状为 ``[sz, sz]`` 的上三角矩阵。
       """

    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):
    """
           初始化位置编码模块。

           参数:
           d_model (int): 模型的隐藏层维度。
           dropout (float): Dropout比率，默认为0.1。
           max_len (int): 最大序列长度，默认为5000。
           """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)