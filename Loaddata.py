import torchvision
from torchvision import transforms
import torch
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_data = torchvision.datasets.CIFAR10(root='./CIFAR10data',train=True,
            download=True,transform=transform )
# 用于打乱数据集，每次以不同数据返回（这里好像还有不少坑，但是我还没亲自掉里面过，
# 不过迟早的事，关于shuffle的坑以后有机会在进行详细阐述）
#当dataloader加载数据时，一次性创建num_workers个工作进程，并用
# batch_sampler将指定batch分配给指定worker，worker将它负责的batch加载进RAM。
train_loader = torch.utils.data.DataLoader(train_data,batch_size = 4,
            shuffle = True,num_workers=2)
test_data = torchvision.datasets.CIFAR10(root='./CIFAR10data',train=False,
            download=True,transform=transform)
test_loader = torch.utils.data.DataLoader(test_data,batch_size = 4,
            shuffle = False,num_workers=2)