import torch
import numpy as np
import sys
import d2lzh_pytorch as d21
sys.path.append("..")

batch_size = 256   # 设置批量大小为256
train_iter, test_iter = d21.load_data_fashion_mnist(batch_size)

# 初始化模型参数
num_inputs = 784    # 使用向量表示每个样本。图像的长和高是28，所以输入向量的长度为28*28=784，向量中的每个元素对应图像中的每个像素
num_outputs = 10    # 图像10个类别，输出层的输出个数为10个

# 权重和偏差参数分别为187*10，1*10的矩阵
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs,num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

# 模型参数梯度
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

'''
#对多维tensor按维度操作，对同一行（dim=0）同一列（dim=1）求和并在结果中保留行和列的维度（keepdim=true）
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))
print(X.sum(dim=1, keepdim=True))  
'''
# 定义函数

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1,keepdim=True)
    return X_exp/partition
'''
#test
x=torch.rand((2,5))
x_prob=softmax(x)
print(x_prob,x_prob.sum(dim=1))
'''

# 定义回归模型，view函数将每张原始图片改成长度为num_inputs的向量
def net(X):
    return softmax(torch.mm(X.view((-1,num_inputs)),W)+b)

# gather函数，y_hat是两个样本在三个类别中的预测概率，y是这里两个样本的标签，使用gather函数得到相关的预测高铝
y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
y=torch.LongTensor([0,2])
print(y_hat.gather(1,y.view(-1,1)))


# 定义softmax回归使用的交叉熵损失函数
def cross_entropy(y_hat,y):
    return - torch.log(y_hat.gather(1,y.view(-1,1)))

# 定义准确率函数，y_hat.argmax(dim=1)返回矩阵y_hat每行中的最大元素的索引
def accuracy(y_hat,y):
    return (y_hat.argmax(dim=1)==y).float().mean().item()

print (accuracy(y_hat,y))

# 类似，我们评价模型net在数据集上data_iter的准确率，准确率应该接近类别个数的倒数0.1
def evaluate_accuracy(data_iter, net):
    acc_sum, n=0.0,0
    for X, y in data_iter:
        acc_sum+=(net(X).argmax(dim=1)==y).float().sum().item()
        n+=y.shape[0]
    return acc_sum/n
print(evaluate_accuracy(test_iter,net))

# 训练模型，迭代周期epochs和学习率lr，（超参数）
num_epochs, lr = 5, 0.1
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X,y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            #梯度清零
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
train_ch3(net,train_iter,test_iter,cross_entropy,num_epochs,batch_size,[W,b],lr)

# 预测。给定一系列的图像（第三行），比较真实结果和预测结果（第一，第二行输出）

X, y = iter(test_iter).next()

true_labels = d21.get_fashion_mnist_labels(y.numpy())
pred_labels = d21.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())

titles = [true +'\n'+ pred for true,pred in zip(true_labels,pred_labels)]

d21.show_fashion_mnist(X[0:9],titles[0:9])