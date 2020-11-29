'''
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
'''

import time
import torch
from torch import nn, optim
import d2lzh_pytorch as d2l


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 6, 5), nn.Sigmoid(),
                                  nn.MaxPool2d(2, 2), nn.Conv2d(6, 16, 5),
                                  nn.Sigmoid(), nn.MaxPool2d(2, 2))
        self.fc = nn.Sequential(nn.Linear(16 * 4 * 4, 120), nn.Sigmoid(),
                                nn.Linear(120, 84), nn.Sigmoid(),
                                nn.Linear(84, 10))

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


net = LeNet()
print(net)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)


def evaluate_accuracy(
    data_iter,
    net,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(
                    dim=1) == y.to(device)).float().sum().cpu().item()
            else:
                if ('is_training' in net.__code__.co_varnames):
                    acc_sum += (net(X, is_training=False).argmax(
                        dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(
                        dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device,
              num_epochs):
    net = net.to(device)
    print("trainint on ", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print(
            'epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
               test_acc, time.time() - start))


lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
