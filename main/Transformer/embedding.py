import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 定义超级简单的词汇表
word_list = ['1', '2', '3']

# 构建词嵌入层
embedding_dim = 3  # 嵌入向量的维度
vocab_size = len(word_list)
embedding = nn.Embedding(vocab_size, embedding_dim)

# 定义输入
input_ids = torch.tensor([0, 1, 2])  # 对应词汇表中的 '1', '2', '3'

# 获取词向量
word_vectors = embedding(input_ids)

# 将词向量的内容提取出来
word_vectors_numpy = word_vectors.detach().numpy()

# 将词向量内容乘以维度的开方
word_vectors_sqrt = word_vectors * embedding_dim**0.5
word_vectors_sqrt_numpy = word_vectors_sqrt.detach().numpy()


# 绘制词向量和处理后的词向量
plt.figure()

# 绘制原始词向量
for i in range(len(word_list)):
    plt.plot([0, word_vectors_numpy[i][0]], [0, word_vectors_numpy[i][1]], linestyle='-', marker='o', markersize=8, label='Original '+word_list[i])

# 绘制处理后的词向量
for i in range(len(word_list)):
    plt.plot([0, word_vectors_sqrt_numpy[i][0]], [0, word_vectors_sqrt_numpy[i][1]], linestyle='-', marker='o', markersize=8, label='Scaled '+word_list[i])

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Word Vectors Comparison')
plt.grid(True)
plt.legend()
plt.show()
