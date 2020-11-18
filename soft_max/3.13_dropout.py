import torch
import torch.nn as nn
import numpy as np
import d2lzh_pytorch as d2l
import sys
sys.path.append("..")

# 将以drop——prob的概率丢弃x中的元素
def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1   # assert断言，判别一个表达式，当表达式错误的时候会触发异常
    keep_prob = 1 - drop_prob

    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return torch.zeros_like(X)  # zeros_like()生成和括号内变量维度维度一致的全是零的内容。
    mask = (torch.randn(X.shape) < keep_prob).float()

    return mask * X / keep_prob


num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

w1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens1)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens1,requires_grad=True)
w2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens1, num_hiddens2)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_hiddens2, requires_grad=True)
w3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens2, num_outputs)), dtype=torch.float, requires_grad=True)
b3 = torch.zeros(num_outputs,requires_grad=True)

params = [w1, b1, w2, b2, w3, b3]


# 定义模型
drop_prob1, drop_prob2 =0.2, 0.5

def net(x, is_training=True):
    x = x.view(-1, num_inputs)
    H1 = (torch.matmul(x, w1) + b1).relu()
    if is_training:  # 只在训练时使用丢弃法
        H1 = dropout(H1, drop_prob1)   # 在第一层全连接后添加丢弃层
    H2 = (torch.matmul(H1, w2) + b2).relu()  # matul函数张量的乘法
    if is_training:
        H2 = dropout(H2,drop_prob2)    # 在第二层全连接后添加丢弃层
    return torch.matmul(H2, w3) + b3


def evaluate_accuracy(data_iter,net):
    acc_sum,n=0.0,0
    for x,y in data_iter:
        if isinstance(net, torch.nn.Module):
            net.eval()  # 评估模式，这回关闭dropout
            acc_sum += (net(x).argmax(dim=1) == y).float().sumn().item()
            net.train()  # 改回训练模式
        else:   # 自定义模型
            if('is_training' in net.__code__.co_varnames):  # 如果有is——training这个参数就将其设置为false
                acc_sum += (net(x, is_training=False).argmax(dim=1) == y).float().sum().item()
            else:
                acc_sum += (net(x).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


# 训练和测试模型

num_epochs, lr, batch_size = 5, 100.0, 256
loss = torch.nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
