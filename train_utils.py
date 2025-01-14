# train_utils.py

import torch
import torch.optim as optim
from data_loader import trainloader_mnist, testloader_mnist, trainloader_cifar, testloader_cifar
from models import LeNet5, WideResNet
from sam_optimizer import SAM
from train_eval import train, evaluate, train_with_sam, save_plots
from datetime import datetime
import os

def log_message(message):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open("training_log.txt", "a") as log_file:
        log_file.write(f"{current_time} - {message}\n")
    print(f"{current_time} - {message}")

def save_model(model, optimizer_name, dataset_name, epoch):
    # 创建模型保存路径
    save_dir = f"model/{dataset_name}/{optimizer_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 保存模型参数
    model_path = f"{save_dir}/epoch_{epoch}.pth"
    torch.save(model.state_dict(), model_path)
    log_message(f"Model saved to {model_path}")

def train_and_evaluate(dataset_name, model, trainloader, testloader, optimizer_sgd, optimizer_adam, optimizer_sam, num_epochs, device):
    sgd_train_losses, sgd_test_losses = [], []
    sgd_train_accuracies, sgd_test_accuracies = [], []
    adam_train_losses, adam_test_losses = [], []
    adam_train_accuracies, adam_test_accuracies = [], []
    sam_train_losses, sam_test_losses = [], []
    sam_train_accuracies, sam_test_accuracies = [], []

    # 记录学习率和权重更新
    sgd_lr_list, adam_lr_list, sam_lr_list = [], [], []
    sgd_weight_updates, adam_weight_updates, sam_weight_updates = [], [], []

    for epoch in range(num_epochs):
        log_message(f'Epoch {epoch + 1}/{num_epochs} - {dataset_name}')

        # SGD
        train_loss_sgd, train_accuracy_sgd, sgd_lr, sgd_weight_update = train(model, trainloader, optimizer_sgd, device)
        test_loss_sgd, test_accuracy_sgd = evaluate(model, testloader, device)
        sgd_train_losses.append(train_loss_sgd)
        sgd_test_losses.append(test_loss_sgd)
        sgd_train_accuracies.append(train_accuracy_sgd)
        sgd_test_accuracies.append(test_accuracy_sgd)
        sgd_lr_list.extend(sgd_lr)
        sgd_weight_updates.extend(sgd_weight_update)
        log_message(f'SGD - Train Loss: {train_loss_sgd:.4f}, Test Loss: {test_loss_sgd:.4f}, Train Accuracy: {train_accuracy_sgd:.2f}%, Test Accuracy: {test_accuracy_sgd:.2f}%')
        save_model(model, "sgd", dataset_name, epoch + 1)

        # Adam
        train_loss_adam, train_accuracy_adam, adam_lr, adam_weight_update = train(model, trainloader, optimizer_adam, device)
        test_loss_adam, test_accuracy_adam = evaluate(model, testloader, device)
        adam_train_losses.append(train_loss_adam)
        adam_test_losses.append(test_loss_adam)
        adam_train_accuracies.append(train_accuracy_adam)
        adam_test_accuracies.append(test_accuracy_adam)
        adam_lr_list.extend(adam_lr)
        adam_weight_updates.extend(adam_weight_update)
        log_message(f'Adam - Train Loss: {train_loss_adam:.4f}, Test Loss: {test_loss_adam:.4f}, Train Accuracy: {train_accuracy_adam:.2f}%, Test Accuracy: {test_accuracy_adam:.2f}%')
        save_model(model, "adam", dataset_name, epoch + 1)

        # SAM
        train_loss_sam, train_accuracy_sam, _, sam_lr, sam_weight_update = train_with_sam(model, trainloader, optimizer_sam, device)
        test_loss_sam, test_accuracy_sam = evaluate(model, testloader, device)
        sam_train_losses.append(train_loss_sam)
        sam_test_losses.append(test_loss_sam)
        sam_train_accuracies.append(train_accuracy_sam)
        sam_test_accuracies.append(test_accuracy_sam)
        sam_lr_list.extend(sam_lr)
        sam_weight_updates.extend(sam_weight_update)
        log_message(f'SAM - Train Loss: {train_loss_sam:.4f}, Test Loss: {test_loss_sam:.4f}, Train Accuracy: {train_accuracy_sam:.2f}%, Test Accuracy: {test_accuracy_sam:.2f}%')
        save_model(model, "sam", dataset_name, epoch + 1)

    # 保存损失和准确率曲线
    save_plots(sgd_train_losses, sgd_test_losses, sgd_train_accuracies, sgd_test_accuracies,
               adam_train_losses, adam_test_losses, adam_train_accuracies, adam_test_accuracies,
               sam_train_losses, sam_test_losses, sam_train_accuracies, sam_test_accuracies,
               sgd_lr_list, adam_lr_list, sam_lr_list,
               sgd_weight_updates, adam_weight_updates, sam_weight_updates,
               dataset_name)