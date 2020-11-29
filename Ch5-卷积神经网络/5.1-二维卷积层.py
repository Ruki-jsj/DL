import torch
from torch import nn

def corr2d(x, k):
    h, w = k.shape
    y = torch.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (x[i: i + h, j: j+w] * k).sum()
    return y

x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
k = torch.tensor([[0, 1], [2, 3]])

print(corr2d(x, k))


# 基于corr2d函数来实现一个自定义的二维卷积层，
# 在构造——init--里声明weight和bias这两个参数模型。前向计算函数则是直接调用corr2d函数在加上偏差

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

# 应用：检测图像中物体的边缘， 也就是找到像素变化的位置，
# 首先我们构造一张6*8的图像也即是高和宽为6像素和8像素的图像，其中中间4列为黑（0），其余为白
x =torch.ones(6, 8)
x[:, 2:6] = 0
print(x)

# 构造一个高和宽分别为1和2的卷积核k，做与输入相互关运算时，如果横向相邻元素相同，就输出0，否则就输出非0
k = torch.tensor([[1, -1]])

# 下面将输入的x和我们设计的卷积核k做相互关运算。
# 可以看出来，我们将从白到黑的边缘和从黑到白的边缘分别检测成了1和-1.否则输出为非0

y = corr2d(x,k)
print(y)

# 卷积层可以通过重复使用卷积核有效的表征空间



# 使用物体边缘检测中的输入变量x和输出数据y来学习我们构造的核数组k
# 先构造一个卷积层，将被初始化为随机数组，在每一次迭代中，使用平方误差来比较y和卷积层的输出，计算梯度来更新权重

# 构造一个核数组的shape（1， 2）的二维卷积层
conv2d = Conv2D(kernel_size=(1, 2))

step = 20
lr = 0.01
for i in range(step):
    y_hat = conv2d(x)
    l = ((y_hat - y) ** 2).sum()
    l.backward()

    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    # 梯度清零
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    if (i + 1) % 5 == 0:
        print('step %d, loss %.3f' % (i + 1, l.item()))


print("weight:", conv2d.weight.data)  # 与我们之前定义的参数差不多
print("bias:", conv2d.bias.data)  # 偏置参数接近0