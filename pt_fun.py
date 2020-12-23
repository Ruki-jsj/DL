import torch
from torch import nn


# js
# print(torch.cuda.is_available())

# 返回一个张量，包含了(0,1)的均匀分布中抽取的随机数组
# print(torch.rand(2,3))
# 返回一个张量，包含了从标准正态分布（均值为0，方差为1，即高斯白噪声）中抽取的一组随机数

x = torch.tensor([[0, 2],[3,4],[9,8]])
print(x.shape)

a = torch.ones(5, 3, 4)
b = torch.ones(4, 2)
print(a)
print(b)
c = torch.matmul(a, b)
print(c.shape)

