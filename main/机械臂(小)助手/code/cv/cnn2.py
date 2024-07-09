import torch
import torch.nn.functional as F
import time

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 读取图像文件
image_path = './img/Camera1_2024-07-08_14-34-24.png'  # 替换为你的图像文件路径
image = Image.open(image_path)

# 将图像转换为numpy数组
image_array = np.array(image)


# 如果需要，可以查看图像数组的形状和内容
print("Image shape:", image_array.shape)
print("Image data:\n", image_array)

# tensor = torch.tensor(image_array)
# # 向上移动内容，最后一行使用矩阵的最后一行
# moved_matrix_u = torch.cat((tensor[1:, :, :], tensor[-1:, :, :]), dim=0)
# moved_matrix_d = torch.cat((tensor[:1, :, :], tensor[:-1, :, :]), dim=0)
# moved_matrix_l = torch.cat((tensor[:, 1:, :], tensor[:, -1:, :]), dim=1)
# moved_matrix_r = torch.cat((tensor[:, :1, :], tensor[:, :-1, :]), dim=1)
# m_u = torch.sum(torch.sub(tensor, moved_matrix_u), dim=2)
# m_d = torch.sum(torch.sub(tensor, moved_matrix_d).pow(2), dim=2)
# m_l = torch.sum(torch.sub(tensor, moved_matrix_l).pow(2), dim=2)
# m_r = torch.sum(torch.sub(tensor, moved_matrix_r).pow(2), dim=2)
#
# m_sub = torch.sub(tensor, moved_matrix_u)
# min = m_sub.min()
# max = m_sub.max()
# tensor_normalized = (tensor - min) / (max - min)  # 归一化到 0-1
#
# m_temp1 = torch.sub(tensor, moved_matrix_u).pow(2)
# m_temp1[:,:,3] = 255
# m_temp2 = torch.sub(tensor, moved_matrix_d)
# m_temp2[:,:,3] = 255
# m_temp3 = torch.sub(tensor, moved_matrix_l)
# m_temp3[:,:,3] = 255
# m_temp4 = torch.sub(tensor, moved_matrix_r)
# m_temp4[:,:,3] = 255
#
# print(m_temp1)
#
# fig, axs = plt.subplots(2, 2, figsize=(30,30))
#
#
# # 将NumPy数组转换为PIL图像
# img1 = Image.fromarray(tensor_normalized.numpy())
# img2 = Image.fromarray(m_temp2.numpy())
# img3 = Image.fromarray(m_temp3.numpy())
# img4 = Image.fromarray(m_temp4.numpy())
#
# # 在子图中展示图片
# axs[0][0].imshow(img1)
# axs[0][1].imshow(img2)
# axs[1][0].imshow(img3)
# axs[1][1].imshow(img4)
#
# # 显示图形
# plt.show()

# image = Image.fromarray(m_temp.numpy())  # 转换为PIL图像格式
# image.show()


# m = torch.stack((m_u, m_d, m_l, m_r), dim=2)
# m, _ = torch.max(m, dim=2)


# image = Image.fromarray(m_u.numpy(), mode="L")  # 转换为PIL图像格式
# image.show()

# tensor1 = torch.cat((tensor[1:, :, :], tensor[-1:, :, :]), dim=0)
# tensor1 = torch.cat((tensor[:1, :, :], tensor[:-1, :, :]), dim=0)
# print(tensor1)
#
# tensor = torch.tensor([
#     [[1, 2, 3], [3, 4, 5]],
#     [[2, 3, 4], [5, 6, 7]]
#
# ])
# # tensor = torch.randn(5, 6, 3)
# print(tensor)
# sum_tensor = torch.sum(tensor, dim=2)
# print(sum_tensor)
