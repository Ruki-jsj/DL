import torch
from torch import nn

def pool2d(X, pool_size, mode='max'):
    X = X.float()
    p_h, p_w = pool_size
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w +1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j +  p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(pool2d(X, (2, 2)))
print(pool2d(X, (2,2), 'avg'))

# 填充和步幅下
x = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))  # 构造一个（1， 1， 4， 4）的输入数据，前两个维度是批量和通道

print(x)
pool2d = nn.MaxPool2d(3) # 下面使用3*3的池化窗口，默认获得形状（3， 3）的步幅
print(pool2d(x))

# 我们可以⼿手动指定步幅和填充。
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(x))

# 也可以指定⾮非正⽅方形的池化窗⼝口，并分别指定⾼高和宽上的填充和步幅。
pool2d = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
print(pool2d(x))

# 多通道
# 在处理多通道输⼊数据时，池化层对每个输入通道分别池化，⽽不是像卷积层那样将各通道的输⼊按通道相加。
# 这意味着池化层的输出通道数与输入通道数相等。下面将数组 X 和 X+1 在通道维上连结来构造通道数为2的输⼊。

x = torch.cat((x, x+1),dim=1)
print(x)

# 池化后，发现通道数还是2
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(x))

# 最大池化和平均池化分别取池化窗口中输⼊元素的最大值和平均值作为输出。池化层的⼀个主要作⽤是缓解卷积层对位置的过度敏感性。
# 可以指定池化层的填充和步幅。
# 池化层的输出通道数跟输入通道数相同。
