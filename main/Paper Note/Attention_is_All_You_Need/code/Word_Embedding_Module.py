import torch
from torch import nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embeddings, self).__init__()
        # lookup table: 查找表，类似新华字典的作用，保存每个词的向量信息
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

vocab_size = 10  # 词汇表大小（词典也就是查找表中一共有10个词）
embedding_dim = 5  # 嵌入维度(一般为512)

model = Embeddings(vocab_size, embedding_dim)

input_cab = torch.tensor([1,2,3])
output = model(input_cab)
print(output)