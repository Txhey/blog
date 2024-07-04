import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# Define inputs
query = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
key = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
value = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]])
mask = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
test = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])

import torch

# 假设的 QKV 张量，形状为 (batch_size, seq_length, hidden_dim)
QKV = torch.randn(2, 3, 6)

print(QKV)
# 定义头的数量
num_heads = 2

# 计算 head_dim
head_dim = QKV.size(-1) // num_heads

# 验证 hidden_dim 是否能被 num_heads 整除
assert QKV.size(-1) % num_heads == 0, "hidden_dim must be divisible by num_heads"

# 变换形状为 (batch_size, seq_length, num_heads, head_dim)
QKV = QKV.view(QKV.size(0), QKV.size(1), num_heads, head_dim)
print(QKV.shape)

# 变换形状为 (batch_size, num_heads, seq_length, head_dim)
QKV = QKV.permute(0, 2, 1, 3)

print(QKV)
print(QKV.shape)  # 输出形状为 (2, 2, 3, 3)


# Compute attention
# output, attn_weights = attention(query, key, value, mask=mask)

# print("Output:", output)
# print("Attention Weights:", attn_weights)
