import torch

# 创建一个形状为 (2, 3) 的张量
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print("Original Tensor:")
print(x)

# 转置张量
x_t = x.transpose(0, 1)
print("\nTransposed Tensor:")
print(x_t)

# 尝试直接对转置后的张量进行 view 操作
try:
    x_t_view = x_t.view(6)
except RuntimeError as e:
    print("\nError when trying to view transposed tensor:")
    print(e)

# 使用 contiguous() 使张量连续
x_t_cont = x_t.contiguous()
print(x_t_cont)

# 对连续的张量进行 view 操作
x_t_cont_view = x_t_cont.view(6)
print("\nContiguous and Viewed Tensor:")
print(x_t_cont_view)
