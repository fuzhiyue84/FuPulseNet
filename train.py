from torch import optim
import torch.nn as nn
from model.cnn import Net
from model.AlexNet import AlexNet
from model.VGG import VGG
from torch.utils.data import DataLoader
from dataset import Pulse
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.models as models

# 调用模型
net= models.resnet50(pretrained=True)
fc_features = net.fc.in_features
net.fc = nn.Linear(fc_features, 4)
#net = net
print(net)
dataset = Pulse('/input/data')
dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=0, drop_last=False)
#定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.09)

for epoch in range(100):

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):

        # 输入数据
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        # 梯度清零
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # 更新参数
        optimizer.step()

        # 打印log信息
        running_loss += loss.data[0]
        if i % 20 == 19:  
            print('[%d, %5d] loss: %.3f' \
                  % (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0
print('Finished Training')
