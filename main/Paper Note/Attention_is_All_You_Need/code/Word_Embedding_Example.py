import matplotlib.pyplot as plt
import numpy as np

# 定义词向量和颜色
vectors = {
    'THH': np.array([1, 4]),
    '***': np.array([2, 4]),
    'Male': np.array([3, 1]),
    'Female': np.array([4, 1]),
    'Guizhou University': np.array([4, 4]),
    'School': np.array([4, 5])
}
colors = {
    'THH': 'r',
    '***': 'g',
    'Male': 'b',
    'Female': 'c',
    'Guizhou University': 'm',
    'School': 'y'
}

# 设置图形
fig, ax = plt.subplots()

# 绘制向量
for word, vec in vectors.items():
    ax.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1, color=colors[word])
    ax.text(vec[0], vec[1], word, size=10, zorder=1, color=colors[word])

# 设置轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')

# 设置图形范围
ax.set_xlim([0, 6])
ax.set_ylim([0, 6])

# 显示图形
plt.grid()
plt.show()
