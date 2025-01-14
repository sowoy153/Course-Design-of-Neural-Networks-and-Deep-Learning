# main_optimized.py

import torch
from train_utils_optimized import train_and_evaluate_sam_compare, log_message
from data_loader import trainloader_mnist, testloader_mnist, trainloader_cifar, testloader_cifar
from models import LeNet5, WideResNet
from sam_optimizer import SAM
from sampro import SAMOptimized
import torch.optim as optim

def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")

    # MNIST
    model_mnist = LeNet5().to(device)
    optimizer_sam_mnist = SAM(model_mnist.parameters(), optim.SGD, rho=0.05, lr=0.05)
    optimizer_sam_optimized_mnist = SAMOptimized(model_mnist.parameters(), optim.SGD, rho=0.05, adaptive_rho=True, max_grad_norm=1.0, lr=0.05)

    # CIFAR-10
    model_cifar = WideResNet().to(device)
    optimizer_sam_cifar = SAM(model_cifar.parameters(), optim.SGD, rho=0.05, lr=0.05)
    optimizer_sam_optimized_cifar = SAMOptimized(model_cifar.parameters(), optim.SGD, rho=0.05, adaptive_rho=True, max_grad_norm=1.0, lr=0.05)

    num_epochs = 40

    # 训练和评估CIFAR-10
    train_and_evaluate_sam_compare("CIFAR-10", model_cifar, trainloader_cifar, testloader_cifar, optimizer_sam_cifar, optimizer_sam_optimized_cifar, num_epochs, device)

    # 训练和评估MNIST
    train_and_evaluate_sam_compare("MNIST", model_mnist, trainloader_mnist, testloader_mnist, optimizer_sam_mnist, optimizer_sam_optimized_mnist, num_epochs, device)

if __name__ == '__main__':
    log_message("Training started (SAM vs SAM Optimized)")
    main()
    log_message("Training completed (SAM vs SAM Optimized)")