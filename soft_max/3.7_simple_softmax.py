import torch
from torch import nn
from torch.nn import init
import d2lzh_pytorch as d21
import sys
sys.path.append("..")

# 设置批量大小
batch_size = 256
train_iter, test_iter = d21.load_data_fashion_mnist(batch_size)

# softmax输出是一个全连接层，用一个线性模块就行
num_inputs = 784
num_outputs = 10


class LinearNet(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(LinearNet,self).__init__()
        self.linear=nn.Linear(num_inputs,num_outputs)

    def forward(self, x):
        y=self.linear(x.view(x.shape[0],-1))  # 数据返回的每个batch样本x的形状为（batch——size，1，28，28）使用view函数将形状转化为（batch_size，784）
        return y


net = LinearNet(num_inputs,num_outputs)


# x的形状转化函数
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


# 定义模型
from collections import OrderedDict
net = nn.Sequential(OrderedDict([('flatten', FlattenLayer()), ('linear', nn.Linear(num_inputs, num_outputs))]))

# 初始化权重参数，均值为0，标准差为0.01
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# 一个稳定的交叉熵函数
loss = nn.CrossEntropyLoss()

# 定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

# 训练模型
num_epochs = 5
d21.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,optimizer)