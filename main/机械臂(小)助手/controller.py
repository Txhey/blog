import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
import os


def funcList(x):
    t = torch.stack(
        [x, torch.sin(x), torch.cos(x), torch.log(torch.abs(x) + 1e-6), torch.exp(x),
         torch.pow(x, torch.e)]).view(-1)
    return torch.nan_to_num(t, nan=0.0)


class ArmNet(nn.Module):
    def __init__(self):
        super(ArmNet, self).__init__()
        n = int(3 * math.pow(6, 3))
        self.l0 = nn.Linear(3, 3)
        self.l1 = nn.Linear(18, 18)
        self.l2 = nn.Linear(108, 108)
        self.linear = nn.Linear(108, 3)

    def forward(self, x):
        x = self.l0(x)
        x = funcList(x)
        x = self.l1(x)
        x = funcList(x)
        x = self.l2(x)
        # # print("中间层1数据：", x)
        # x = funcList(x)
        # # print("中间层2数据：", x)
        # x = funcList(x)
        # # print("中间层3数据：", x)
        x = self.linear(x)
        # # print("最终 x:", x)
        return x


model = ArmNet()
if os.path.exists('model.pth'):
    model.load_state_dict(torch.load('model.pth'))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

num = 0

losses = []


def train_step(input_angles, target_position):
    model.train()
    optimizer.zero_grad()

    input_angles_nparray = np.array(input_angles)
    input_angles_nparray.astype(np.float32)
    input_angles_nparray /= 360
    input_angles_tensor = torch.tensor(input_angles_nparray, dtype=torch.float32).requires_grad_(True)
    target_position_tensor = torch.tensor(target_position, dtype=torch.float32).requires_grad_(True)

    y_hat = model(input_angles_tensor)
    loss = criterion(y_hat, target_position_tensor)
    loss.backward()
    optimizer.step()

    global num
    num += 1

    # 收集损失值并更新图表
    losses.append(loss.item())

    # 打印调试信息
    if (num % 100 == 0):
        print(f'Epoch [{num}], Loss: {loss.item()}')
        print("Input Angles:", input_angles_tensor)
        print("Target Position:", target_position_tensor)
        print("y_hat:", y_hat)

        # 保存模型的状态字典
        torch.save(model.state_dict(), 'model.pth')

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.ylim(0, np.max(losses[-100:]))
        plt.plot(losses, 'b')
        plt.show()
    return loss.item(), y_hat


x = torch.tensor([1.0, 2.0, 3.0])
y = train_step(x, [2.0, 3.0, 4.0])
print(y)

from flask import Flask, request, jsonify
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    # print("data:", data)
    angles = data['angles']
    target_position = data['targetPos']
    # 进行一次训练步骤
    loss, y_hat = train_step(angles, target_position)
    # print(y_hat)
    # 返回训练损失
    j = jsonify({'loss': loss, 'x': float(y_hat[0]), 'y': float(y_hat[1]), 'z': float(y_hat[2])})
    # print(j)
    return j


if __name__ == '__main__':
    # 禁用调试模式
    app.run(port=5000)

plt.ioff()
