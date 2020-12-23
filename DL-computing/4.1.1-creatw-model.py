import torch
from torch import nn


# model类是nn模块的模型构造类，所有神经网络的基类
# 定义MLP类重载了model类的__init__函数和forward函数分别用于创建模型和定义前向传播

class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类block的构造函数来进行必要的初始化，这样构造的实例可以指定其他的函数
        # 参数params
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256) # 隐藏层
        self.act =  nn.ReLU()
        self.output = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向计算，也就是根据x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)
# MLP类不需要定义反向传播函数，系统将通过自动求梯度而自动生成反向传播所需要的backward函数


x = torch.rand(2, 784)
net = MLP()
print(net)
print(net(x))

