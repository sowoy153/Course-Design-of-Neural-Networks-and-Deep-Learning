import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

def train(model, trainloader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    weight_updates = []  # 记录权重更新
    lr_list = []  # 记录学习率

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        # 记录权重更新
        weight_update = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        weight_updates.append(weight_update)

        # 记录学习率
        lr_list.append(optimizer.param_groups[0]['lr'])

        running_loss += loss.item()

        # 计算训练准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(trainloader)
    train_accuracy = 100 * correct / total
    return train_loss, train_accuracy, lr_list, weight_updates

def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss = running_loss / len(testloader)
    test_accuracy = 100 * correct / total
    return test_loss, test_accuracy

def train_with_sam(model, trainloader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    gradient_norms = []  # 记录梯度范数
    weight_updates = []  # 记录权重更新
    lr_list = []  # 记录学习率

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        # 记录梯度范数
        grad_norm = optimizer._grad_norm()
        gradient_norms.append(grad_norm.item())

        # 记录权重更新
        weight_update = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        weight_updates.append(weight_update)

        # 记录学习率
        lr_list.append(optimizer.get_lr())

        optimizer.first_step(zero_grad=True)

        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.second_step(zero_grad=True)

        running_loss += loss.item()

        # 计算训练准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(trainloader)
    train_accuracy = 100 * correct / total
    return train_loss, train_accuracy, gradient_norms, lr_list, weight_updates

def save_plots(sgd_train_losses, sgd_test_losses, sgd_train_accuracies, sgd_test_accuracies,
               adam_train_losses, adam_test_losses, adam_train_accuracies, adam_test_accuracies,
               sam_train_losses, sam_test_losses, sam_train_accuracies, sam_test_accuracies,
               sgd_lr_list, adam_lr_list, sam_lr_list,  # 学习率
               sgd_weight_updates, adam_weight_updates, sam_weight_updates,  # 权重更新
               dataset_name):
    # 创建 image 文件夹
    if not os.path.exists("image"):
        os.makedirs("image")

    # 绘制损失曲线
    plt.figure()
    plt.plot(sgd_train_losses, label="SGD Train Loss", color="blue", linestyle="-")
    plt.plot(sgd_test_losses, label="SGD Test Loss", color="blue", linestyle="--")
    plt.plot(adam_train_losses, label="Adam Train Loss", color="green", linestyle="-")
    plt.plot(adam_test_losses, label="Adam Test Loss", color="green", linestyle="--")
    plt.plot(sam_train_losses, label="SAM Train Loss", color="red", linestyle="-")
    plt.plot(sam_test_losses, label="SAM Test Loss", color="red", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{dataset_name} - Loss Curve")
    plt.legend()
    plt.savefig(f"image/{dataset_name}_loss.png")
    plt.close()

    # 绘制准确率曲线
    plt.figure()
    plt.plot(sgd_train_accuracies, label="SGD Train Accuracy", color="blue", linestyle="-")
    plt.plot(sgd_test_accuracies, label="SGD Test Accuracy", color="blue", linestyle="--")
    plt.plot(adam_train_accuracies, label="Adam Train Accuracy", color="green", linestyle="-")
    plt.plot(adam_test_accuracies, label="Adam Test Accuracy", color="green", linestyle="--")
    plt.plot(sam_train_accuracies, label="SAM Train Accuracy", color="red", linestyle="-")
    plt.plot(sam_test_accuracies, label="SAM Test Accuracy", color="red", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{dataset_name} - Accuracy Curve")
    plt.legend()
    plt.savefig(f"image/{dataset_name}_accuracy.png")
    plt.close()

    # 绘制学习率曲线
    plt.figure()
    plt.plot(sgd_lr_list, label="SGD Learning Rate", color="blue", linestyle="-")
    plt.plot(adam_lr_list, label="Adam Learning Rate", color="green", linestyle="-")
    plt.plot(sam_lr_list, label="SAM Learning Rate", color="red", linestyle="-")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title(f"{dataset_name} - Learning Rate Curve")
    plt.legend()
    plt.savefig(f"image/{dataset_name}_learning_rate.png")
    plt.close()

    # 绘制权重更新曲线
    plt.figure()
    plt.plot(sgd_weight_updates, label="SGD Weight Updates", color="blue", linestyle="-")
    plt.plot(adam_weight_updates, label="Adam Weight Updates", color="green", linestyle="-")
    plt.plot(sam_weight_updates, label="SAM Weight Updates", color="red", linestyle="-")
    plt.xlabel("Epoch")
    plt.ylabel("Weight Updates")
    plt.title(f"{dataset_name} - Weight Update Curve")
    plt.legend()
    plt.savefig(f"image/{dataset_name}_weight_updates.png")
    plt.close()
def train_with_sam_optimized(model, trainloader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    gradient_norms = []  # 记录梯度范数
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        
        # 记录梯度范数
        grad_norm = optimizer._grad_norm()
        gradient_norms.append(grad_norm.item())
        
        optimizer.first_step(zero_grad=True)
        
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.second_step(zero_grad=True)
        
        running_loss += loss.item()

        # 计算训练准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(trainloader)
    train_accuracy = 100 * correct / total
    return train_loss, train_accuracy, gradient_norms

def train_with_sam_hessian(model, trainloader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    gradient_norms = []  # 记录梯度范数
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        
        # 记录梯度范数
        grad_norm = optimizer._grad_norm()
        gradient_norms.append(grad_norm.item())
        
        optimizer.first_step(zero_grad=True)
        
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.second_step(zero_grad=True)
        
        running_loss += loss.item()

        # 计算训练准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(trainloader)
    train_accuracy = 100 * correct / total
    return train_loss, train_accuracy, gradient_norms