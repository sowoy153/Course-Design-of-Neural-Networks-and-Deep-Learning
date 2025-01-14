import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models import WideResNet
import torch.nn.functional as F
import os

# Grad-CAM 实现
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def hook_fn(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(hook_fn)

    def generate_cam(self, input_image, target_class=None):
        # 前向传播
        model_output = self.model(input_image)
        if target_class is None:
            target_class = torch.argmax(model_output, dim=1).item()

        # 反向传播
        self.model.zero_grad()
        one_hot_output = torch.zeros_like(model_output)
        one_hot_output[0][target_class] = 1
        model_output.backward(gradient=one_hot_output, retain_graph=True)

        # 计算加权特征图
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(pooled_gradients * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # 应用ReLU激活函数
        cam = F.interpolate(cam, size=input_image.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam.squeeze().detach().cpu().numpy()  # 使用 detach() 分离张量

    def visualize(self, input_image, target_class=None, save_path=None):
        cam = self.generate_cam(input_image, target_class)
        input_image = input_image.squeeze().permute(1, 2, 0).cpu().numpy()
        input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())

        # 叠加热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam_image = heatmap + np.float32(input_image)
        cam_image = cam_image / np.max(cam_image)

        # 显示结果
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(input_image)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Grad-CAM")
        plt.imshow(cam_image)
        plt.axis('off')

        # 保存图像
        if save_path:
            plt.savefig(save_path)
        plt.close()

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=True)

# 定义模型路径
model_paths = {
    "Adam": r"model\CIFAR-10\adam\epoch_40.pth",
    "SGD": r"model\CIFAR-10\sgd\epoch_40.pth",
    "SAM": r"model\CIFAR-10\sam\epoch_40.pth",
    "SAM Optimized": r"model_optimized\CIFAR-10\sam_optimized\epoch_40.pth",
    "SAM Hessian": r"model_pro_sam\CIFAR-10\sam_hessian\epoch_40.pth"  # 添加SAM Hessian的模型路径
}

# 创建保存图像的目录
output_dir = "gradcam_results"
os.makedirs(output_dir, exist_ok=True)

# 固定随机种子，确保每次运行时使用相同的图像
torch.manual_seed(42)
np.random.seed(42)

# 选择固定的图像
fixed_images = []
fixed_labels = []
for i, (images, labels) in enumerate(testloader):
    fixed_images.append(images)
    fixed_labels.append(labels)
    if i >= 4:  # 只选择 5 张图像
        break

# 加载模型并生成 Grad-CAM 热力图
for optimizer_name, model_path in model_paths.items():
    # 加载模型
    model = WideResNet(depth=10, widen_factor=4, num_classes=10)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 设置目标层为 block3 的最后一个卷积层
    target_layer = model.block3[-3]  # 最后一个卷积层
    print(f"Target layer for {optimizer_name}: {target_layer}")

    # 创建 Grad-CAM 对象
    grad_cam = GradCAM(model, target_layer)

    # 对每个固定图像生成 Grad-CAM 热力图
    for i, (image, label) in enumerate(zip(fixed_images, fixed_labels)):
        save_path = os.path.join(output_dir, f"{optimizer_name}_image_{i}.png")
        grad_cam.visualize(image, target_class=label.item(), save_path=save_path)
        print(f"Saved Grad-CAM for {optimizer_name} - Image {i} to {save_path}")