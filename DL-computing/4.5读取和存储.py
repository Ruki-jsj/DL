import torch
from torch import nn

# 创建tensor张量x，将其存储到x.pt文件中
x = torch.ones(3)
torch.save(x, 'x.pt')

# 从存储文件读取回内存
x2 = torch.load('x.pt')
print(x2)

# 创建一个tensor列表并读取回内存
y = torch.zeros(4)
torch.save([x, y], 'xy.pt')
xy_list = torch.load('xy.pt')
print(xy_list)

# 存储并且读取一个字符串映射到tensor的字典
torch.save({'x':x, 'y':y}, 'xy_dict.pt')
xy = torch.load('xy_dict.pt')
print(xy)

# 4.5.2.1 读写模型-state_dict（从一个参数名称隐射到参数tensor的字典对象）
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)


    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

net = MLP()
print(net.state_dict())

# 注意，只有具有学习参数的层才有state_dict中的条目。
# 优化器（optim）也有一个state_dict，其中包含关于优化器状态以及使用的超参数的信息

optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)  #  momentum动量
print(optimizer.state_dict())

# 4.5.2.2保存和加载模型
# pytorch中保存和加载模型的有两种常见的方法：



# 保存和加载state-dict(推荐)
# 保存 torch.save(net2.state_dict(), PATH)
# 加载,model = net2(*args, **kwargs),model.load_state_dict(torch.load(PATH))

X = torch.randn(2, 3)
Y = net(X)

PATH = './net.pt'
torch.save(net.state_dict(), PATH)

net2 = MLP()
net2.load_state_dict(torch.load(PATH))
Y2 = net2(X)

print(Y2 == Y)

# 2保存和加载整个模型
# 保存：torch.save(model, PATH) 加载：model = torch.load（PATH）




