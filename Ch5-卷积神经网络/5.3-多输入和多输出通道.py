# 5.3.1多输入通道
# 实现瀚多个输入通道数的互相关运算，只需对每个通道做互相关运算，然后通过add_n函数进行相加

import torch
from torch import nn
import sys
sys.path.append("..")
import d2lzh_pytorch as d21

def corr2d_multi_in(X, K):
    res = d21.corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += d21.corr2d(X[i, :, :], K[i, :, :])
    return res

X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                 [[1, 2, 3], [4, 5, 6], [7, 8, 8]]])
K = torch.tensor([[[0, 1], [2, 3]],
                  [[1, 2], [3, 4]]])
print(corr2d_multi_in(X,K))


# 5.3.2多输出通道
# 实现一个互相关运算函数来计算多个通道的输出
def corr2d_multi_in_out(X,K):
    # 对K的第0维遍历，每次同输入x做互相关计算。所有结果使用stack函数合并在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K])

# 将核数组K同K+1(k中每个元素加1)和K+2连结在一起构造处一个输出通道数为3的卷积核
K = torch.stack([K, K+1, K+2])
print(K.shape)  #torch.Size([3, 2, 2, 2])

# 对输入数组X与核数组K做互相关运算。此时的输出函数有三个通道。
# 其中第一个通道的结果与之前输入数组X与多输入通道，单输出通道核的计算结果一致

print(corr2d_multi_in_out(X,K))


# 5.3.1 1*1卷积层
def corr2d_multi_in_out_1X1(X,K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i, h * w)
    K = K.view(c_o, c_i)
    Y = torch.mm(K, X) # 全连接层的矩阵乘法
    return Y.view(c_o, h, w)

X = torch.rand(3, 3, 3)
K = torch.rand(2, 3, 1, 1)

Y1 = corr2d_multi_in_out_1X1(X, K)
Y2 = corr2d_multi_in_out(X, K)

print((Y1 - Y2).norm().item() < 1e-6)