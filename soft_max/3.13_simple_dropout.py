import torch
import torch.nn as nn
import numpy as np
import d2lzh_pytorch as d2l
import sys
sys.path.append("..")

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

w1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens1)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens1,requires_grad=True)
w2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens1, num_hiddens2)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_hiddens2, requires_grad=True)
w3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens2, num_outputs)), dtype=torch.float, requires_grad=True)
b3 = torch.zeros(num_outputs,requires_grad=True)

params = [w1, b1, w2, b2, w3, b3]

drop_prob1, drop_prob2 =0.2, 0.5

net = nn.Sequential(d2l.FlattenLayer(),
                    nn.Linear(num_inputs, num_hiddens1),
                    nn.ReLU(), nn.Dropout(drop_prob1),
                    nn.Linear(num_hiddens1, num_hiddens2),
                    nn.ReLU(), nn.Dropout(drop_prob2),
                    nn.Linear(num_hiddens2, 10))

for param in net.parameters():
    nn.init.normal_(param, mean=0, std=0.01)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
num_epochs, batch_size = 5, 256
loss = torch.nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
