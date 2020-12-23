import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from IPython import display
from matplotlib import pyplot as plt
import random
import torch
import time
import d2lzh_pytorch as d21
import torchvision
import torch.utils.data
import sys
from torch import nn
sys.path.append("..")
import torchvision.transforms as transforms


def use_svg_display():
    # 用矢量图表示
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


# 按batch—size读取数据，每次返回batch-size个随机样本的特征和标签
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


# 将数值标签转换为相应的文本标签
def get_fashion_mnist_labels(labels):
    text_labels=['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
    return [text_labels[int(i)] for i in labels]



# 一行里可以画出多张图像和对应标签的函数
def show_fashion_mnist(images,labels):
    d21.use_svg_display()
    _,figs=plt.subplots(1,len(images),figsize=(12,12))
    for f,img,lbl, in zip(figs,images,labels):
        f.imshow(img.view((28,28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


def load_data_fashion_mnist(batch_size):
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=False,transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=False,transform=transforms.ToTensor())
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter


def sgd(params,lr,batch_size):
    for param in params:
        param.data-=lr*param.grad/batch_size


def evaluate_accuracy(data_iter,net):
    acc_sum,n=0.0,0
    for x,y in data_iter:
        acc_sum+=(net(x).argmax(dim=1)==y).float().sum().item()
        n+=y.shape[0]
    return acc_sum/n


# x的形状转化
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()

    def forward(self,x):
        return x.view(x.shape[0],-1)

# 训练模型
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X,y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            if optimizer is None:
                d21.sgd(params,lr,batch_size)
            else:
                optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) ==y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'% (epoch + 1, train_l_sum / n, train_acc_sum / n,test_acc))

    def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
        d21.set_figsize(figsize)
        d21.plt.xlabel(x_label)
        d21.plt.ylabel(y_label)
        d21.plt.semilogy(x_vals, y_vals)
        if x2_vals and y2_vals:
            d21.plt.semilogy(x2_vals, y2_vals, linestyle=':')
            d21.plt.legend(legend)

# 定义模型
def linreg(X, w, b):
    return torch.mm(X, w) + b

# 定义损失函数，平方损失最为损失函数
def squard_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 定义，训练和测试模型，先定义作图函数semilogy，y轴使用了对数尺度
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals = None, y2_vals = None,legend = None,figsize = (3.5, 2.5)):
    d21.set_figsize(figsize)
    d21.plt.xlabel(x_label)
    d21.plt.ylabel(y_label)
    d21.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d21.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d21.plt.legend(legend)
    plt.show()

def evaluate_accuracy1(data_iter,net):
    acc_sum,n=0.0,0
    for x,y in data_iter:
        if isinstance(net, torch.nn.Module):
            net.eval()
            acc_sum += (net(x).argmax(dim=1) == y).float().sumn().item()
            net.train()
        else:
            if('is_training' in net.__code__.co_varnames):
                acc_sum += (net(x, is_training=False).argmax(dim=1) == y).float().sum().item()
            else:
                acc_sum += (net(x).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n



# 二维互相关运算
def corr2d(x, k):
    h, w = k.shape
    y = torch.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (x[i: i + h, j: j+w] * k).sum()
    return y

# 5.5.1lenet模型所使用的evaluate—accuracy函数的改进
def evaluate_accuracy5(
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
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec' % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
