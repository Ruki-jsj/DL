import torch
import torch.utils.data
import numpy as np
import d2lzh_pytorch as d21
import sys
from matplotlib import pyplot as plt
sys.path.append("..")

# 生成数据集，生成一个人工的数据集，使用三阶多项式函数来生成样本的标签
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5

# torch.randn()返回一个张量，包含了从标准正态分布（均值为0，方差为1，即高斯白噪声）中抽取的一组随机数
features = torch.randn((n_train+n_test, 1))

# torch.cat是将两个张量（tensor）拼接在一起，cat是concatnate的意思，即拼接，联系在一起,最后的1为一维行（列）
# torch,pow对输入的每分量求幂次运算
poly_feature = torch.cat((features, torch.pow(features, 2),torch.pow(features, 3)), 1)

# 函数解析式,X[:,0]就是取所有行的第0个数据, X[:,1] 就是取所有行的第1个数据,X[1,:]就是取所有列的第1个数据,
labels = (true_w[0]*poly_feature[:, 0]+true_w[1]*poly_feature[:, 1]+true_w[2]*poly_feature[:, 2]+true_b) # 广播机制

# 加上噪声项，尾项一个正态分布均值为0，标准差为0.01
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)


# 查看前两个数据
print(features[:2], poly_feature[:2], labels[:2])


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

# 多项式的拟合训练和测试，将模型的定义部分放在fit and plot
num_epochs, loss = 100, torch.nn.MSELoss()


def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = torch.nn.Linear(train_features.shape[-1], 1)  # 全连接框架，对于train_features.shape[-1]输入，1个输出

    batch_size = min(10, train_labels.shape[0])

    # 包装数据和目标张量的数据集
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)

    #数据加载器。组合数据集和采样器，并在数据集上提供单进程或多进程迭代器，样本数据，样本标签（目标）
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01) # 优化算法SGD
    train_ls, test_ls = [], []
    for _ in  range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1,1))
            optimizer.zero_grad()   # 梯度清零
            l.backward()
            optimizer.step()  #更新所有的参数
        train_labels = train_labels.view(-1,1)
        test_labels = test_labels.view(-1,1)
        train_ls.append(loss(net(train_features), train_labels).item())  # append()函数用于在列表末尾添加新的对象，修改原本的列表
        test_ls.append(loss(net(test_features),test_labels).item())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls,'epoch', 'loss', range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data, '\nbias:', net.bias.data)


# 三阶多项式函数拟合（正常）训练出来的模型参数值接近参数值：w1=1.2, w2=-3.4, w3=5.6, b=5
fit_and_plot(poly_feature[:n_train,:], poly_feature[n_train:,:], labels[:n_train], labels[n_train:])


# 线性函数的拟合（欠拟合）
fit_and_plot(features[:n_train,:], features[n_train:,:], labels[:n_train], labels[n_train:])

# 训练样本不足（过拟合）
fit_and_plot(poly_feature[0:2, :], poly_feature[n_train:, :], labels[0:2], labels[n_train:])


'''
由于无法从训练误差估计泛化误差，一味的降低训练误差并不意味这泛化误差会降低
机器学习的模型应该关注降低泛化误差
欠拟合函数无法得到较低的训练误差，过拟合指模型的训练误差远小于他在测试集上的误差
所以应该选择复杂度合适的模型并且避免使用过少的训练样本
'''

