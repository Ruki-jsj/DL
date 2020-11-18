import torch
from torch import nn
from torch.nn import init
import numpy as np
import d2lzh_pytorch as d21
import sys
sys.path.append("..")

# 定义模型，设置hiddens为256,将relu作为激活函数
num_inputs, num_outputs, num_hiddens =784, 10, 256

net = nn.Sequential(d21.FlattenLayer(), nn.Linear(num_inputs,num_hiddens), nn.ReLU(), nn.Linear(num_hiddens,num_outputs))
for params in net.parameters():
    init.normal_(params,mean=0,std=0.01)

# 读取数据并训练模型
batch_size = 256
train_iter, test_iter = d21.load_data_fashion_mnist(batch_size)
loss=torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5
d21.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)