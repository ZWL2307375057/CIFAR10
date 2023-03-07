# import torchvision
# from torchvision import transforms
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.autograd import Variable
#
# #  1.Load data
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
# train_data = torchvision.datasets.CIFAR10(root='./CIFAR10data',train=True,
#             download=True,transform=transform )
# # 用于打乱数据集，每次以不同数据返回（这里好像还有不少坑，但是我还没亲自掉里面过，
# # 不过迟早的事，关于shuffle的坑以后有机会在进行详细阐述）
# #当dataloader加载数据时，一次性创建num_workers个工作进程，并用
# # batch_sampler将指定batch分配给指定worker，worker将它负责的batch加载进RAM。
# train_loader = torch.utils.data.DataLoader(train_data,batch_size = 4,
#             shuffle = True,num_workers=2)
# test_data = torchvision.datasets.CIFAR10(root='./CIFAR10data',train=False,
#             download=True,transform=transform)
# test_loader = torch.utils.data.DataLoader(test_data,batch_size = 4,
#             shuffle = False,num_workers=2)
#
# # 2.Build Model
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3,6,5)
#         self.pool = nn.MaxPool2d(2,2)
#         self.conv2 = nn.Conv2d(6,16,5)
#         self.fc1 = nn.Linear(16*5*5,120)
#         self.fc2 = nn.Linear(120,84)
#         self.fc3 = nn.Linear(84,10)
#     def forward(self,x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1,16*5*5)  #拉成向量
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         return x
#
# # 3.Train
# net  = Net()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(),lr = 0.001,momentum=0.9) #SGD(传入参数，学习率，动量）
#
# for epoch in range(1):
#     running_loss = 0.0
#     # 0 用于指定索引起始值
#     for i, data in enumerate(train_loader, 0):
#         input, target = data
#         input, target = Variable(input), Variable(target)
#         optimizer.zero_grad()
#         output = net(input)
#         loss = criterion(output, target)  # out 和target的交叉熵损失
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.data
#
#         if i % 2000 == 1999:  ## print every 2000 mini_batches,1999,because of index from 0 on
#             print('[%d,%5d]loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0
#     print('Finished Training')
#


import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 导入数据并及进行标准化处理，转换成需要的格式

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 下载数据
train_data = torchvision.datasets.CIFAR10(root='./CIFAR10data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)
test_data = torchvision.datasets.CIFAR10(root='./CIFAR10data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    # np.transpose :按需求转置
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 定义卷积神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # 拉成向量
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


net = Net()
# 定义loss函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # SGD(传入参数，学习率，动量）
# 训练网络
if __name__ == '__main__':

    for epoch in range(1):
        running_loss = 0.0
        # 0 用于指定索引起始值
        for i, data in enumerate(train_loader, 0):
            input, target = data
            input, target = Variable(input), Variable(target)
            optimizer.zero_grad()
            output = net(input)
            loss = criterion(output, target)  # out 和target的交叉熵损失
            loss.backward()
            optimizer.step()

            running_loss += loss.data

            if i % 2000 == 1999:  ## print every 2000 mini_batches,1999,because of index from 0 on
                print('[%d,%5d]loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        print('Finished Training')

        dataier = iter(test_loader)
        print( '############')
        images, labels = next(dataier)
        imshow(torchvision.utils.make_grid(images))
        print('GroundTruth:', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
        print('%%%%%%%%')
        outputs = net(Variable(images))
        _, pred = torch.max(outputs.data, 1)
        print('Predicted: ', ' '.join('%5s' % classes[pred[j]] for j in range(4)))

        correct = 0.0
        total = 0
        for data in test_loader:
            images, labels = data
            outputs = net(Variable(images))
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum()
        print('Accuracy of the network on the 10000 test images :%d %%' % (100 * correct / total))

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        for data in test_loader:
            images, labels = data
            outputs = net(Variable(images))
            _, pred = torch.max(outputs.data, 1)
            c = (pred == labels).squeeze()  # 1*10000*10-->10*10000
            for i in range(4):
                label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
