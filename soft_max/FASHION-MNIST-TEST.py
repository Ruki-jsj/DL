import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
import time
import sys
import d2lzh_pytorch as d21
import torch.utils.data
import numpy
sys.path.append("..")


#mnist_train,mnist_test都是torch。ytils.data.Dataset的子类

mnist_train=torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',train=True,download=True,transform=transforms.ToTensor())
mnist_test=torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',train=False,download=True,transform=transforms.ToTensor())
print(type(mnist_train))
print(len(mnist_train),len(mnist_test))     #len()获取数据集的大小,训练集和测试集分别为60000和10000

feature,lable=mnist_train[0]    #feature是（C*H*w），对应高和宽均为28像素的图像
print(feature.shape,lable)      #第一维是通道数，灰度图像，通道数为1，后面两位为高，宽

#将数值标签转换为相应的文本标签
def get_fashion_mnist_labels(labels):
    text_labels=['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
    return [text_labels[int(i)] for i in labels]

#一行里可以画出多张图像和对应标签的函数
def show_fashion_mnist(images,labels):
    d21.use_svg_display()
    _,figs=plt.subplots(1,len(images),figsize=(12,12))
    for f,img,lbl, in zip(figs,images,labels):
        f.imshow(img.view((28,28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

X,y=[],[]
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))

#读取小批量数据
batch_size=256
if sys.platform.startswith('win'):
    num_workers=0   #不使用额外的进程来加速读取数据
else:
    num_workers=4   #设置4个进程来读取数据
train_iter=torch.utils.data.DataLoader(mnist_train,batch_size=batch_size,shuffle=False,num_workers=num_workers)
test_iter=torch.utils.data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False,num_workers=num_workers)

#读取一遍训练数据的时间
start=time.time()
for X,y in train_iter:
    continue
print('%.2f sec'%(time.time()-start))

'''
softmax回归适用于分类问题，通过softmax运算出类别的概率分布，
它是个单层神经网络
fashion-mnist是个十类分类数据集
'''