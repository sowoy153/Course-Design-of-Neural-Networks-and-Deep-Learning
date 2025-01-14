# main.py

import torch
from train_utils import train_and_evaluate
from train_utils_optimized import train_and_evaluate_sam_compare
from train_utils_SAMHessian import train_and_evaluate_sam_compare as train_and_evaluate_sam_compare_hessian
from data_loader import trainloader_mnist, testloader_mnist, trainloader_cifar, testloader_cifar, trainloader_fashion_mnist, testloader_fashion_mnist
from models import LeNet5, WideResNet
from sam_optimizer import SAM
from sampro import SAMOptimized
from SAMHessian import SAMHessian
import torch.optim as optim
from train_utils import log_message

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MNIST
    model_mnist = LeNet5().to(device)
    optimizer_sgd_mnist = optim.SGD(model_mnist.parameters(), lr=0.01)
    optimizer_adam_mnist = optim.Adam(model_mnist.parameters(), lr=0.001)
    optimizer_sam_mnist = SAM(model_mnist.parameters(), optim.SGD, rho=0.05, lr=0.05)
    optimizer_sam_optimized_mnist = SAMOptimized(model_mnist.parameters(), optim.SGD, rho=0.05, adaptive_rho=True, max_grad_norm=1.0, lr=0.05)
    optimizer_sam_hessian_mnist = SAMHessian(model_mnist.parameters(), optim.SGD, rho=0.05, adaptive_rho=True, max_grad_norm=1.0, hessian_lambda=0.01, lr=0.05)

    # CIFAR-10
    model_cifar = WideResNet().to(device)
    optimizer_sgd_cifar = optim.SGD(model_cifar.parameters(), lr=0.01)
    optimizer_adam_cifar = optim.Adam(model_cifar.parameters(), lr=0.001)
    optimizer_sam_cifar = SAM(model_cifar.parameters(), optim.SGD, rho=0.05, lr=0.05)
    optimizer_sam_optimized_cifar = SAMOptimized(model_cifar.parameters(), optim.SGD, rho=0.05, adaptive_rho=True, max_grad_norm=1.0, lr=0.05)
    optimizer_sam_hessian_cifar = SAMHessian(model_cifar.parameters(), optim.SGD, rho=0.05, adaptive_rho=True, max_grad_norm=1.0, hessian_lambda=0.01, lr=0.05)

    # Fashion MNIST
    model_fashion_mnist = LeNet5().to(device)
    optimizer_sam_fashion_mnist = SAM(model_fashion_mnist.parameters(), optim.SGD, rho=0.05, lr=0.05)
    optimizer_sam_optimized_fashion_mnist = SAMOptimized(model_fashion_mnist.parameters(), optim.SGD, rho=0.05, adaptive_rho=True, max_grad_norm=1.0, lr=0.05)
    optimizer_sam_hessian_fashion_mnist = SAMHessian(model_fashion_mnist.parameters(), optim.SGD, rho=0.05, adaptive_rho=True, max_grad_norm=1.0, hessian_lambda=0.01, lr=0.05)

    num_epochs = 40

    # 训练和评估CIFAR-10
    train_and_evaluate("CIFAR-10", model_cifar, trainloader_cifar, testloader_cifar, optimizer_sgd_cifar, optimizer_adam_cifar, optimizer_sam_cifar, num_epochs, device)
    train_and_evaluate_sam_compare("CIFAR-10", model_cifar, trainloader_cifar, testloader_cifar, optimizer_sam_cifar, optimizer_sam_optimized_cifar, num_epochs, device)
    train_and_evaluate_sam_compare_hessian("CIFAR-10", model_cifar, trainloader_cifar, testloader_cifar, optimizer_sam_cifar, optimizer_sam_optimized_cifar, optimizer_sam_hessian_cifar, num_epochs, device)

    # 训练和评估MNIST
    train_and_evaluate("MNIST", model_mnist, trainloader_mnist, testloader_mnist, optimizer_sgd_mnist, optimizer_adam_mnist, optimizer_sam_mnist, num_epochs, device)
    train_and_evaluate_sam_compare("MNIST", model_mnist, trainloader_mnist, testloader_mnist, optimizer_sam_mnist, optimizer_sam_optimized_mnist, num_epochs, device)
    train_and_evaluate_sam_compare_hessian("MNIST", model_mnist, trainloader_mnist, testloader_mnist, optimizer_sam_mnist, optimizer_sam_optimized_mnist, optimizer_sam_hessian_mnist, num_epochs, device)

    # 训练和评估Fashion MNIST
    train_and_evaluate_sam_compare_hessian("Fashion MNIST", model_fashion_mnist, trainloader_fashion_mnist, testloader_fashion_mnist, optimizer_sam_fashion_mnist, optimizer_sam_optimized_fashion_mnist, optimizer_sam_hessian_fashion_mnist, num_epochs, device)

if __name__ == '__main__':
    log_message("Training started")
    main()
    log_message("Training completed")