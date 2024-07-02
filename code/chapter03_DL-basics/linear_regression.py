import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.utils.data as Data
import numpy as np

torch.manual_seed(1)
# 创建数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(
    0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01,
                       size=labels.size()), dtype=torch.float)

# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)

# 把 dataset 放入 DataLoader
data_iter = Data.DataLoader(
    dataset=dataset,      # torch TensorDataset format
    batch_size=10,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
)

# 定义模型
model = nn.Sequential(
    nn.Linear(num_inputs, 1),
)
print(model)

# 初始化模型
nn.init.normal_(model[0].weight, mean=0.0, std=0.01)
nn.init.constant_(model[0].bias, val=0.0)

lossfunc = nn.MSELoss()  # 使用MSE损失
optimizer = optim.Adam(model.parameters(), lr=0.1)  # 使用Adam优化器
scheduler = CosineAnnealingLR(optimizer, T_max=10)  # 使用余弦下降的学习率


for epoch in range(1, 11):
    model.train()  # 设置模型为训练模式
    for data, label in data_iter:
        output = model(data)
        loss = lossfunc(output, label.view(-1, 1))
        optimizer.zero_grad()  # 将梯度清零
        loss.backward()
        optimizer.step()

    scheduler.step()
    print(f'Epoch {epoch}, Loss: {loss.item():.8}')  # 打印每个 epoch 的损失


dense = model[0]
print(true_w, dense.weight.item())
print(true_b, dense.bias.item())
