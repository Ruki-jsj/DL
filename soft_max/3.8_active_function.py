import matplotlib.pyplot as plt
import torch
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d21


def xyplot(x_vals, y_vals, name):
    d21.set_figsize(figsize=(5,2.5))
    d21.plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    d21.plt.xlabel('x')
    d21.plt.ylabel(name + 'x')
    plt.show()


x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)

y = x.relu()  # relu函数--max（x，0）
xyplot(x, y, 'relu')

# 对relu函数求导
y.sum().backward()
xyplot(x, x.grad, 'grad of relu')

# sigmoid函数0--1/（1+exp（-x））
y = x.sigmoid()
xyplot(x, y, 'sigmoid')

# 对sigmoid求导
x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of sigmoid')

# tanh函数--[1-exp（-2x）]/[1+exp（-2x）]
y=x.tanh()
xyplot(x, y, 'tanh')

# 求导
x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of tanh')
'''
多层感知机在输出层和输入层之间加入了一个或者多个全连接隐藏层，并且通过激活函数对隐藏层
输出进行变化，以上是常用的激活函数
'''
