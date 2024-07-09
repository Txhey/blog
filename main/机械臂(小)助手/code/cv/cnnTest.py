import torch
import torch.nn.functional as F
import time

# 定义 6x6 矩阵，内容从 0 到 35
matrix = torch.arange(36, dtype=torch.float32).view(6, 6)
print("Original Matrix:")
print(matrix)
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.6f} seconds")
        return result
    return wrapper


@measure_time
def reshape_matrices(base_matrix, image_matrix):
    # 将 base_matrix 转换为卷积核
    base_matrix = base_matrix.view(1, 1, 2, 2)

    # 将 image_matrix 转换为 4D 张量 (batch_size, channels, height, width)
    image_matrix = image_matrix.view(1, 1, 6, 6)

    return base_matrix, image_matrix


@measure_time
def subtract_and_square(base_matrix, image_matrix):
    # 使用 unfold 展开 image_matrix，使其适应相减操作
    unfolded = F.unfold(image_matrix, kernel_size=(2, 2))
    unfolded = unfolded.view(1, 1, 4, -1)
    print(unfolded)
    print(unfolded)
    print(base_matrix.view(1,1,4,1))
    # 相减并取平方
    output = (unfolded - base_matrix.view(1, 1, 4, 1)) ** 2
    output = output.sum(dim=2).view(1, 1, 5, 5)  # 计算平方和并重塑为 4x4 矩阵

    # 移除多余的维度
    output = output.squeeze()

    return output


# 定义 base_matrix 和 image_matrix
base_matrix = torch.tensor([
    [0.0, 1.0],
    [6.0, 7.0],
])

image_matrix = matrix

# 重塑矩阵
base_matrix, image_matrix = reshape_matrices(base_matrix, image_matrix)

# 进行相减和取平方
output = subtract_and_square(base_matrix, image_matrix)

# 展示 output 的形状
print(f"Output Matrix Shape: {output.shape}")

# 如果需要输出值
print(output)
