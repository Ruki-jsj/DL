import torch.nn as nn
import torch
import torch.utils.data
import pandas as pd
# pandas库读入并且处理数据
import d2lzh_pytorch as d21
import sys
sys.path.append("..")

# print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

train_data = pd.read_csv('../../data/kaggle_house/train.csv')
test_data = pd.read_csv('../../data/kaggle_house/test.csv')

# print(train_data.shape)  # 训练集包括1460个样本，80个特征和1个标签
# print(test_data.shape)   # 测试机包括1459个样本，80个特征，我们需要将测试集中的每个样本的标签预测出来
# print(train_data.iloc[0:4, [0, 1, 2, -3, -2, -1]])  # 查看前四个特征，后两个特征和标签

# 将所有的训练数据和测试数据的79个特征按样本连结
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))


# 预处理数据，对连续的数据做标准化处理，对于缺失的特征值，将其替换成该特征的均值
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda  x: (x - x.mean()) / (x.std()))
# 标准化后， 每个特征的均值变为0，所以可以直接用0来替换缺失值
all_features = all_features.fillna(0)

# 将离散数值转成指示特征，dummy-na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
# print(all_features.shape)  # 特征数从79增加到了354


# 通过value属性得到Numpy格式的数据，并且转成NDArry方便以后的训练
n_train =train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)


# 训练模型，使用基本的线性回归模型和平方损失函数来训练模型
loss = torch.nn.MSELoss()


def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net


# 定义比赛用来评价模型的对数均方根误差
def log_rmse(net, features, labels):
    with torch.no_grad():
        # 将小于1的值设置为1，使得取对数时数值更加稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())
    return rmse.item()


# 训练函数与之前有所不同，这次使用了Adam优化算法，相对于之前的梯度下降算法，对学习率不那么敏感
def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_deacy, batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    # 这里使用了Adma优化算法
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_deacy)
    net = net.float()
    for epoch in  range(num_epochs):
        for x, y in train_iter:
            l = loss(net(x.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# 用来选择模型设计并调节超参数，下面的函数返回第i折交叉验证时所需要的训练和验证数据
def get_k_fold_data(k, i, X, y ):
    assert k >1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in  range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid


# 训练k次并返回训练和验证的平均误差
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i  in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d21.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse', range(1, num_epochs + 1), valid_ls, ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k


k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))


def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_deacy, batch_size):
    net = get_net(train_features.shape[1])
    train_ls, _= train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    d21.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./submission.csv', index=False)


train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)

'''
通常需要对真实数据做预处理
可以使用k折交叉验证来选择模型和调节超参数
'''