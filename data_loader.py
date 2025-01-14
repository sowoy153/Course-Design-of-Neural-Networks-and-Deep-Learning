import torch
import torchvision
import torchvision.transforms as transforms

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载MNIST数据集
trainset_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader_mnist = torch.utils.data.DataLoader(trainset_mnist, batch_size=128, shuffle=True)

testset_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader_mnist = torch.utils.data.DataLoader(testset_mnist, batch_size=128, shuffle=False)

# 加载CIFAR-10数据集
transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
trainloader_cifar = torch.utils.data.DataLoader(trainset_cifar, batch_size=128, shuffle=True)

testset_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
testloader_cifar = torch.utils.data.DataLoader(testset_cifar, batch_size=128, shuffle=False)

# 加载Fashion MNIST数据集
trainset_fashion_mnist = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader_fashion_mnist = torch.utils.data.DataLoader(trainset_fashion_mnist, batch_size=128, shuffle=True)

testset_fashion_mnist = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader_fashion_mnist = torch.utils.data.DataLoader(testset_fashion_mnist, batch_size=128, shuffle=False)