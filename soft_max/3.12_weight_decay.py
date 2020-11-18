import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import d2lzh_pytorch as d21
import sys
sys.path.append("..")

n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05

features = torch.randn((n_train + n_test, num_inputs))
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

train_features, test_features = features[:n_train,:], features[n_train:,:]
train_labels, test_labels = labels[:n_train], labels[n_train:]

# 初始化模型参数，并且给每个参数附上梯度
def init_parama():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


# 定义L2的范数惩罚项，这里只惩罚，模型的权重参数
def l2_penalty(w):
    return (w**2).sum() / 2


# 定义训练和测试，与之前不同的是在损失函数的时候加上了惩罚项
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d21.linreg, d21.squard_loss

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)


def fit_and_plot(lambd):
    w, b = init_parama()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for x,y in train_iter:
            # 添加了l2范数惩罚项
            l = loss(net(x, w, b), y) + lambd * l2_penalty(w)
            l = l.sum()

            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            d21.sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
    d21.semilogy(range(1, num_epochs + 1), train_ls, 'epochs' ,'loss', range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', w.norm().item())


# 观察过拟合
fit_and_plot(lambd=0)

fit_and_plot(lambd=3)