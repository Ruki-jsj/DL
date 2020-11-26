import torch
from torch import nn

class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)

        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)


    def forward(self, x):
        x =  self.linear(x)

        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)

        x= self.linear(x)

        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()


x= torch.rand(2, 20)
net = FancyMLP()
print(net)
print(net(x))


class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU())

    def forward(self, x):
        return self.net(x)

net =  nn.Sequential(NestMLP(), nn.Linear(30, 20), FancyMLP())

x = torch.rand(2, 40)
print(net)
print(net(x))

