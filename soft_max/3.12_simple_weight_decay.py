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


batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d21.linreg, d21.squard_loss

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

def fit_and_plot_pytorch(wd):
    # 对权重参数进行衰减，权重名称一般为weight结尾
    net = nn.Linear(num_inputs, 1)
    nn.init.normal_(net.weight, mean=0, std=1)
    nn.init.normal_(net.bias, mean=0, std=1)
    optimizer_w = torch.optim.SGD(params=[net.weight], lr=lr, weight_decay=wd)  #对权重参数进行衰减
    optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr)  #不对偏参数衰减

    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for x, y in train_iter:
            l = loss(net(x), y).mean()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()

            l.backward()

            # 对两个optimizer实例分别带调用step函数，分别对这两个参数进行更新
            optimizer_w.step()
            optimizer_b.step()
        train_ls.append(loss(net(train_features), train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())
    d21.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1,num_epochs + 1), test_ls,['train','tesr'])
    print('L2 norm of w:',net.weight.data.norm().item())

fit_and_plot_pytorch(0)

fit_and_plot_pytorch(3)




'''
权重衰减可以一定程度上缓解过拟合问题
正则化通过为模型损失函数添加惩罚项使学出的模型参数值较⼩小，是应对过拟合的常⽤用⼿手段。
权重衰减等价于L2范数正则化，通常会使学到的权重参数的元素较接近0
权重衰减可以通过优化器中的 weight_decay 超参数来指定。    
可以定义多个优化器器实例例对不不同的模型参数使⽤用不不同的迭代⽅方法。

'''
