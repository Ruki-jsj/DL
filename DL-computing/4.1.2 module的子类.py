import torch
from torch import nn
# 实现一个与sequential类相同功能的MYsequential类
from collections import OrderedDict

class MYsequential(nn.Module):


    def __init__(self, *args):

        super(MYsequential, self).__init__()

        if len(args) == 1 and isinstance(args[0], OrderedDict):   # 如果传入的是一个orderedDict
            for key, module in args[0].items():
                self.add_module(key, module)
                # add—module方法会将module添加到self._modules（一个ordereddict）

        else:  # 传入一些module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)


    def forward(self, input):
            # self._modules返回一个ordereddict，保证会按照成员添加时候的顺序遍历
        for module in self._modules.values():
            input = module(input)
        return input


# 我们用mysequential类来实现mlp类，并且使用随机初始化模型做一次前向计算
net = MYsequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

x = torch.rand(2, 784)
print(net)
print(net(x))


# modulelist接受一个子模块作为输入，然后就可以像list进行append和extend
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10))
print(net[-1]) # 索引访问
print(net)


# 接受一个子模块的字典作为输入
net =  nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU()
})

net['output'] = nn.Linear(256, 10)  # 添加
print(net['linear'])  # 访问
print(net.output)
print(net)