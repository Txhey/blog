import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def create_distance_matrix(width, height):
    # 计算中心位置
    center_x, center_y = width // 2, height // 2

    # 创建网格来表示每个像素的坐标
    x_coords = torch.arange(width, dtype=torch.float32)
    y_coords = torch.arange(height, dtype=torch.float32)
    X, Y = torch.meshgrid(x_coords, y_coords, indexing='xy')  # 显式指定 indexing='xy'

    # 计算每个像素到中心位置的欧几里得距离
    distances = (X - center_x)**2 + (Y - center_y)**2

    return distances


image_path = './img/Camera1_2024-07-08_20-52-11.png'  # 替换为你的图像文件路径
image_path = './img/Camera1_2024-07-08_20-52-11.png'  # 替换为你的图像文件路径
image = Image.open(image_path)

image_array = np.array(image)
tensor_image = torch.tensor(image_array)
# 获取图像的宽度和高度
width, height = image.size

# 创建距离矩阵
distance_matrix = create_distance_matrix(width, height)

# 找到张量的最小值和最大值
min_val = distance_matrix.min()
max_val = distance_matrix.max()

# 将张量内容反向映射到 1-255 范围
mapped_tensor = (distance_matrix - min_val) / (max_val - min_val)  # 先将值映射到 0-1 范围
mapped_tensor = 255 - (mapped_tensor * 254)  # 然后反向映射到 1-255 范围
mapped_tensor = torch.floor(mapped_tensor).to(torch.int)
mapped_tensor = mapped_tensor.unsqueeze(-1)

# 展示矩阵
# plt.imshow(mapped_tensor, cmap='gray')
# plt.colorbar()
# plt.show()

print(tensor_image.shape)
print(mapped_tensor)

newImage = torch.cat((tensor_image, mapped_tensor), dim=2)
# newImage = tensor_transposed = newImage.permute(1, 2, 0)
# print(newImage)


plt.imshow(newImage)
plt.colorbar()
plt.show()
