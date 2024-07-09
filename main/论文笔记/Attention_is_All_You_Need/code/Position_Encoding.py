import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        # 全0的矩阵，一共max_len行，d_model列
        pe = torch.zeros(max_len, d_model)
        # shape=(max_len,1)的向量，数据从0到max_len-1 [[0],[1],...,[max_len-1]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 三角函数中每列相同的值： 1/(10000^(i/d_model)) 注：这里使用对数和指数而不是直接使用次方计算，是因为这两个函数在底层更优化，计算速度更快。而且可以避免因为数字过大或过小在浮点数计算中超出范围。
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 给pe增加一个维度，表示批次（max_len, d_model） -> (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # 因为位置嵌入不需要更新参数内容
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


# 示例：如何在Transformer中使用位置编码
d_model = 6  # 特征维度
max_len = 3  # 最大序列长度

# 创建位置编码实例
pos_encoder = PositionalEncoding(d_model, max_len)

# 输入序列
src = torch.rand((2, max_len, d_model))  # 序列长度，batch大小，特征维度

print(src)
# 添加位置编码
src = pos_encoder(src)

print(src)