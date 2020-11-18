import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np

# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0,1,(num_examples,num_inputs)),dtype=torch.float)
labels = true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
# y=X*w+b+kexi，k
labels+=torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float)
# exi为随机噪声，数据集中无意义的干扰

# 读取数据
import torch.utils.data as Data
# pytorch中提供了data包来读取数据
batch_size=10   #随机读取是个数据样本的小批量
# 将训练集的特征和标签组合
dataset=Data.TensorDataset(features,labels)
# 随机读取小批量
data_iter=Data.DataLoader(dataset,batch_size,shuffle=True)
for x,y in data_iter:
    print(x,y)
    break

# 定义模型
import torch.nn as nn
class LinearNet(nn.Module):
    def __init__(self,n_feature):
        super(LinearNet,self).__init__()
        self.linear=nn.Linear(n_feature,1)
    # 定义前向传播
    def forward(self,x):
        y=self.linear(x)
        return y
from collections import OrderedDict
net = nn.Sequential(OrderedDict([('linear', nn.Linear(num_inputs, 1))]))
# sequential是个有序容器，网络层将按照在sequential的顺序传入计算图
print(net)
print(net[0])

for param in net.parameters():
    print(param)   # 查看所有可学习得参数

# 初始化模型参数
from torch.nn import init
init.normal_(net[0].weight,mean=0,std=0.01)
init.constant_(net[0].bias,val=0)

# 定义损失函数
loss=nn.MSELoss()

# 定义优化算法
import torch.optim as optim
optimizer=optim.SGD(net.parameters(),lr=0.03)
print(optimizer)

# 训练模型
num_epocha=3
for epoch in range(1,num_epocha+1):
    for x,y in data_iter:
        output=net(x)
        l=loss(output,y.view(-1,1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d,loss: %f'%(epoch,l.item()))

dense=net[0]
print(true_w,dense.weight)
print(true_b,dense.bias)