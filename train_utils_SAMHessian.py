# train_utils_pro_sam.py

import torch
import torch.optim as optim
from data_loader import trainloader_mnist, testloader_mnist, trainloader_cifar, testloader_cifar, trainloader_fashion_mnist, testloader_fashion_mnist
from models import LeNet5, WideResNet
from sampro import SAMOptimized
from SAMHessian import SAMHessian
from train_eval import evaluate, train_with_sam, train_with_sam_optimized, train_with_sam_hessian
from datetime import datetime
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

def log_message(message):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open("training_log_pro_sam.txt", "a") as log_file:
        log_file.write(f"{current_time} - {message}\n")
    print(f"{current_time} - {message}")

def save_model(model, optimizer_name, dataset_name, epoch):
    save_dir = f"model_pro_sam/{dataset_name}/{optimizer_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_path = f"{save_dir}/epoch_{epoch}.pth"
    torch.save(model.state_dict(), model_path)
    log_message(f"Model saved to {model_path}")

def train_with_sam_hessian(model, trainloader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    gradient_norms = []
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        
        grad_norm = optimizer._grad_norm()
        gradient_norms.append(grad_norm.item())
        
        optimizer.first_step(zero_grad=True)
        
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.second_step(zero_grad=True)
        
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(trainloader)
    train_accuracy = 100 * correct / total
    return train_loss, train_accuracy, gradient_norms

def train_and_evaluate_sam_compare(dataset_name, model, trainloader, testloader, optimizer_sam, optimizer_sam_optimized, optimizer_sam_hessian, num_epochs, device):
    sam_train_losses, sam_test_losses = [], []
    sam_train_accuracies, sam_test_accuracies = [], []
    sam_optimized_train_losses, sam_optimized_test_losses = [], []
    sam_optimized_train_accuracies, sam_optimized_test_accuracies = [], []
    sam_hessian_train_losses, sam_hessian_test_losses = [], []
    sam_hessian_train_accuracies, sam_hessian_test_accuracies = [], []
    sam_gradient_norms, sam_optimized_gradient_norms, sam_hessian_gradient_norms = [], [], []

    for epoch in range(num_epochs):
        log_message(f'Epoch {epoch + 1}/{num_epochs} - {dataset_name}')
        
        # SAM
        train_loss_sam, train_accuracy_sam, grad_norms_sam = train_with_sam(model, trainloader, optimizer_sam, device)
        test_loss_sam, test_accuracy_sam = evaluate(model, testloader, device)
        sam_train_losses.append(train_loss_sam)
        sam_test_losses.append(test_loss_sam)
        sam_train_accuracies.append(train_accuracy_sam)
        sam_test_accuracies.append(test_accuracy_sam)
        sam_gradient_norms.append(sum(grad_norms_sam) / len(grad_norms_sam))
        log_message(f'SAM - Train Loss: {train_loss_sam:.4f}, Test Loss: {test_loss_sam:.4f}, Train Accuracy: {train_accuracy_sam:.2f}%, Test Accuracy: {test_accuracy_sam:.2f}%')
        save_model(model, "sam", dataset_name, epoch + 1)
        
        # SAM Optimized
        train_loss_sam_optimized, train_accuracy_sam_optimized, grad_norms_sam_optimized = train_with_sam_optimized(model, trainloader, optimizer_sam_optimized, device)
        test_loss_sam_optimized, test_accuracy_sam_optimized = evaluate(model, testloader, device)
        sam_optimized_train_losses.append(train_loss_sam_optimized)
        sam_optimized_test_losses.append(test_loss_sam_optimized)
        sam_optimized_train_accuracies.append(train_accuracy_sam_optimized)
        sam_optimized_test_accuracies.append(test_accuracy_sam_optimized)
        sam_optimized_gradient_norms.append(sum(grad_norms_sam_optimized) / len(grad_norms_sam_optimized))
        log_message(f'SAM Optimized - Train Loss: {train_loss_sam_optimized:.4f}, Test Loss: {test_loss_sam_optimized:.4f}, Train Accuracy: {train_accuracy_sam_optimized:.2f}%, Test Accuracy: {test_accuracy_sam_optimized:.2f}%')
        save_model(model, "sam_optimized", dataset_name, epoch + 1)
        
        # SAM Hessian
        train_loss_sam_hessian, train_accuracy_sam_hessian, grad_norms_sam_hessian = train_with_sam_hessian(model, trainloader, optimizer_sam_hessian, device)
        test_loss_sam_hessian, test_accuracy_sam_hessian = evaluate(model, testloader, device)
        sam_hessian_train_losses.append(train_loss_sam_hessian)
        sam_hessian_test_losses.append(test_loss_sam_hessian)
        sam_hessian_train_accuracies.append(train_accuracy_sam_hessian)
        sam_hessian_test_accuracies.append(test_accuracy_sam_hessian)
        sam_hessian_gradient_norms.append(sum(grad_norms_sam_hessian) / len(grad_norms_sam_hessian))
        log_message(f'SAM Hessian - Train Loss: {train_loss_sam_hessian:.4f}, Test Loss: {test_loss_sam_hessian:.4f}, Train Accuracy: {train_accuracy_sam_hessian:.2f}%, Test Accuracy: {test_accuracy_sam_hessian:.2f}%')
        save_model(model, "sam_hessian", dataset_name, epoch + 1)

    # 保存损失和准确率曲线
    save_sam_compare_plots(
        sam_train_losses, sam_test_losses, sam_train_accuracies, sam_test_accuracies,
        sam_optimized_train_losses, sam_optimized_test_losses, sam_optimized_train_accuracies, sam_optimized_test_accuracies,
        sam_hessian_train_losses, sam_hessian_test_losses, sam_hessian_train_accuracies, sam_hessian_test_accuracies,
        sam_gradient_norms, sam_optimized_gradient_norms, sam_hessian_gradient_norms, dataset_name
    )

def save_sam_compare_plots(
    sam_train_losses, sam_test_losses, sam_train_accuracies, sam_test_accuracies,
    sam_optimized_train_losses, sam_optimized_test_losses, sam_optimized_train_accuracies, sam_optimized_test_accuracies,
    sam_hessian_train_losses, sam_hessian_test_losses, sam_hessian_train_accuracies, sam_hessian_test_accuracies,
    sam_gradient_norms, sam_optimized_gradient_norms, sam_hessian_gradient_norms, dataset_name
):
    if not os.path.exists("image_pro_sam"):
        os.makedirs("image_pro_sam")
    
    # 绘制损失曲线
    plt.figure()
    plt.plot(sam_train_losses, label="SAM Train Loss", color="blue", linestyle="-")
    plt.plot(sam_test_losses, label="SAM Test Loss", color="blue", linestyle="--")
    plt.plot(sam_optimized_train_losses, label="SAM Optimized Train Loss", color="green", linestyle="-")
    plt.plot(sam_optimized_test_losses, label="SAM Optimized Test Loss", color="green", linestyle="--")
    plt.plot(sam_hessian_train_losses, label="SAM Hessian Train Loss", color="red", linestyle="-")
    plt.plot(sam_hessian_test_losses, label="SAM Hessian Test Loss", color="red", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{dataset_name} - Loss Curve (SAM vs SAM Optimized vs SAM Hessian)")
    plt.legend()
    plt.savefig(f"image_pro_sam/{dataset_name}_loss.png")
    plt.close()

    # 绘制准确率曲线
    plt.figure()
    plt.plot(sam_train_accuracies, label="SAM Train Accuracy", color="blue", linestyle="-")
    plt.plot(sam_test_accuracies, label="SAM Test Accuracy", color="blue", linestyle="--")
    plt.plot(sam_optimized_train_accuracies, label="SAM Optimized Train Accuracy", color="green", linestyle="-")
    plt.plot(sam_optimized_test_accuracies, label="SAM Optimized Test Accuracy", color="green", linestyle="--")
    plt.plot(sam_hessian_train_accuracies, label="SAM Hessian Train Accuracy", color="red", linestyle="-")
    plt.plot(sam_hessian_test_accuracies, label="SAM Hessian Test Accuracy", color="red", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{dataset_name} - Accuracy Curve (SAM vs SAM Optimized vs SAM Hessian)")
    plt.legend()
    plt.savefig(f"image_pro_sam/{dataset_name}_accuracy.png")
    plt.close()

    # 绘制梯度范数曲线
    plt.figure()
    plt.plot(sam_gradient_norms, label="SAM Gradient Norm", color="blue", linestyle="-")
    plt.plot(sam_optimized_gradient_norms, label="SAM Optimized Gradient Norm", color="green", linestyle="-")
    plt.plot(sam_hessian_gradient_norms, label="SAM Hessian Gradient Norm", color="red", linestyle="-")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.title(f"{dataset_name} - Gradient Norm Curve (SAM vs SAM Optimized vs SAM Hessian)")
    plt.legend()
    plt.savefig(f"image_pro_sam/{dataset_name}_gradient_norm.png")
    plt.close()
    
    