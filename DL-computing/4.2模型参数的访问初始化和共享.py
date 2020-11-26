import torch
from torch import nn
from torch.nn import init

net =  nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))

print(net)
X = torch.rand(2, 4)
Y = net(X).sum()


print(type(net.named_parameters()))
for name, param in net.named_parameters():
    print(name, param.size())

for name, param in net[0].named_parameters():
   print(name, param.size(), type(param))

class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)
    def forward(self, x):
        pass


n = MyModel()
for name, param in n.named_parameters():
    print(name)


weight_0 = list(net[0].parameters())[0]
print(weight_0.data)
print(weight_0.grad)
Y.backward()
print(weight_0.grad)



# 初始化参数模型
for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)


# 使用常数初始化权重参数
for name, param in net.named_parameters():
    if 'bias' in name:
        init.constant_(param, val=0)
        print(name, param.data)


# 自定义初始化方法
def normal_(tensor, mean=0, std=1):
    with torch.no_grad():
        return tensor.normal_(mean, std)

def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()


for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)




# 通过改变这些参数的data来改写模型的参数值同时不会影响梯度
for name, param in net.named_parameters():
    if 'bias' in name:
        param.data += 1
        print(name, param.data)

# 4.2.4共享参数模型
linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear)
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)


# 在内存中，这两个线形层其实是一个对象
print(id(net[0]) == id(net[1]))
print(id(net[0].weight) == id(net[1].weight))

x = torch.ones(1, 1)
y = net(x).sum()
print(y)
y.backward()
print(net[0].weight.grad)