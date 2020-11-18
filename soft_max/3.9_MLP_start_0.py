import torch
import numpy as np
import d2lzh_pytorch as d21
import sys
sys.path.append("..")

# 获取和读取数据，依然使用fashion——mnist,对图像进行分类
batch_size =256
train_iter, test_iter = d21.load_data_fashion_mnist(batch_size)

# 定义模型参数
# 输入个数784，输出10，设置超参数256个
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)

b1 = torch.zeros(num_hiddens, dtype=torch.float)

W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)

b2 = torch.zeros(num_outputs, dtype=float)

params = [W1, W2, b1, b2]

for param in params:
    param.requires_grad_(requires_grad=True)


# 定义激活函数,这边使用基础的max函数实现relu
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))


# 定义模型，使用view函数对每张图改成长度为num_inputs的向量
def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1)+b1)
    return torch.matmul(H, W2)+b2


# 定义损失函数，为了数值的稳定性，直接使用pytorch中的交叉损失函数，对数的负数，越接近0，说明性能好
loss = torch.nn.CrossEntropyLoss()

# 训练模型，设置迭代周期为5，学习率为100
num_epochs, lr = 5, 100
d21.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)