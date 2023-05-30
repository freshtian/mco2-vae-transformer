import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        # 初始化位置编码矩阵，形状为 (max_len, d_model)
        # 计算位置编码时使用log空间，以更好地处理长序列
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        # 根据PyTorch版本设置卷积层的填充方式
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # 定义卷积层，将输入特征维度 c_in 转换为嵌入维度 d_model
        # 卷积核大小为3，填充模式为循环填充（padding_mode='circular'），不包含偏置项（bias=False）
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        # 初始化卷积层的权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        print("Shape before permute:", x.shape)
        # 将输入数据的维度重新排列，从 (batch_size, seq_len, feature_dim) 变为 (batch_size, feature_dim, seq_len)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # 将特征嵌入和位置编码相加
        # self.value_embedding(x) 返回 (batch_size, seq_len, d_model)
        # self.position_embedding(x) 返回 (1, seq_len, d_model)，广播到 (batch_size, seq_len, d_model)
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
