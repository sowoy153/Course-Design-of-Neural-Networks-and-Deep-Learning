# main.py

import torch
from train_utils import train_and_evaluate, log_message
from data_loader import trainloader_mnist, testloader_mnist, trainloader_cifar, testloader_cifar
from models import LeNet5, WideResNet
from sam_optimizer import SAM
import torch.optim as optim

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MNIST
    model_mnist = LeNet5().to(device)
    optimizer_sgd_mnist = optim.SGD(model_mnist.parameters(), lr=0.01)
    optimizer_adam_mnist = optim.Adam(model_mnist.parameters(), lr=0.001)
    optimizer_sam_mnist = SAM(model_mnist.parameters(), optim.SGD, rho=0.05, lr=0.05)

    # CIFAR-10
    model_cifar = WideResNet().to(device)
    optimizer_sgd_cifar = optim.SGD(model_cifar.parameters(), lr=0.01)
    optimizer_adam_cifar = optim.Adam(model_cifar.parameters(), lr=0.001)
    optimizer_sam_cifar = SAM(model_cifar.parameters(), optim.SGD, rho=0.05, lr=0.05)

    num_epochs = 40

    # 训练和评估CIFAR-10
    train_and_evaluate("CIFAR-10", model_cifar, trainloader_cifar, testloader_cifar, optimizer_sgd_cifar, optimizer_adam_cifar, optimizer_sam_cifar, num_epochs, device)

    # 训练和评估MNIST
    train_and_evaluate("MNIST", model_mnist, trainloader_mnist, testloader_mnist, optimizer_sgd_mnist, optimizer_adam_mnist, optimizer_sam_mnist, num_epochs, device)

if __name__ == '__main__':
    log_message("Training started")
    main()
    log_message("Training completed")