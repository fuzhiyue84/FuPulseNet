import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# 定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 两个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=3,
                out_channels=6,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),  # 输入通道数，输出通道数，卷积核
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=12,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=24,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=24,
                out_channels=48,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=48,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))


        self.dense = nn.Sequential(
            nn.Linear(96 * 7 * 7, 128),  # 六次最大池化长宽224*84->3*1
            nn.Linear(128, 4))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        #x = self.conv6(x)

        x = x.view(x.size()[0], -1)

        x = self.dense(x)
        return x





'''
定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.09)
'''