import time
import torch
import torch.utils.data
from torch import nn, optim
import torchvision

import sys
sys.path.append("..")
import d2lzh_pytorch as d21
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 96, 11, 4),
                                  nn.ReLU(),
                                  nn.MaxPool2d(3, 2),
                                  nn.Conv2d(96, 256, 5, 1, 2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(3, 2),
                                  nn.Conv2d(256, 384, 3, 1, 1),
                                  nn.ReLU(),
                                  nn.Conv2d(384, 384, 3, 1, 1),
                                  nn.ReLU(),
                                  nn.Conv2d(384, 256, 3, 1, 1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(3, 2))
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )


    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

net = AlexNet()
print(net)

# 读取数据

def load_data_fashion_mnist_56(batch_size, resize=None, root='~/Datasets/FashionMNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=False, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=False, transform=transform)


    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter,test_iter

batch_size = 1
train_iter, test_iter = load_data_fashion_mnist_56(batch_size, resize=224)

# 训练
lr, num_epochs = 0.001, 1
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d21.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


