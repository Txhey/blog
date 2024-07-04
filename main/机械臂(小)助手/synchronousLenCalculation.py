import math


def calculate_belt_length(C, P, T1, T2):
    # 计算节径
    D1 = (P * T1) / math.pi
    D2 = (P * T2) / math.pi

    # 计算同步带长度
    L = 2 * C + math.pi * (D1 + D2) / 2 + ((D2 - D1) ** 2) / (4 * C)

    return L


# 给定参数
C = 47.5 # 中心距 (mm)
P = 2  # 节距 (mm)
T1 =  20  # 同步轮齿数
T2 = 60

# 计算同步带长度
belt_length = calculate_belt_length(C, P, T1, T2)
print(belt_length)