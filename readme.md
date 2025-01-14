# 项目名称

## 项目简介

本项目旨在通过不同优化器（如 SGD、Adam、SAM 及其变种）对多个数据集（如 MNIST、CIFAR-10、Fashion MNIST）进行训练和评估，比较它们的性能表现。

## 文件结构

项目名称/
├── data_loader.py # 数据加载与预处理
├── models.py # 模型定义
├── sam_optimizer.py # 优化器：实现了 SAM 优化器。
├── sampro.py # 优化器：实现了 SAM 优化器的优化版本。
├── SAMHessian.py # 优化器：实现了 SAM Hessian 优化器。
├── train_eval.py # 训练与评估：包含训练和评估模型的函数。
├── train_utils.py # 训练与评估：包含训练和评估的辅助函数。
├── train_utils_optimized.py # 训练与评估：包含优化版本的训练和评估辅助函数。
├── train_utils_SAMHessian.py # 训练与评估：包含 SAM Hessian 版本的训练和评估辅助函数。
├── main.py # 主脚本：负责训练和评估模型。
├── main_optimized.py # 主脚本：优化版本的主脚本。
├── main_SAMHessian.py # 主脚本：SAM Hessian 版本的主脚本。
├── Grad-CAM.py # 结果可视化：实现了 Grad-CAM，用于可视化模型的注意力区域。
├── training_log.txt # 日志文件：记录训练过程的日志文件。
├── training_log_optimized.txt # 日志文件：记录优化版本的训练过程日志文件。
├── training_log_pro_sam.txt # 日志文件：记录 SAM 优化器变种的训练过程日志文件。
├── model/ # 结果存储：存储训练好的模型。
├── model_optimized/ # 结果存储：存储优化版本的训练模型。
├── model_pro_sam/ # 结果存储：存储 SAM 优化器变种的训练模型。
├── image/ # 结果存储：存储训练过程中的图像结果。
├── image_optimized/ # 结果存储：存储优化版本的图像结果。
├── image_pro_sam/ # 结果存储：存储 SAM 优化器变种的图像结果。
├── gradcam_results/ # 结果存储：存储 Grad-CAM 的可视化结果。
└── requirements.txt # 依赖文件：列出项目所需的 Python 包。

## 文件说明

### 数据加载与预处理

- `data_loader.py`：包含加载和预处理 MNIST 和 CIFAR-10 数据集的代码。

### 模型定义

- `models.py`：定义了用于训练的模型结构，如 LeNet5 和 WideResNet。

### 优化器

- `sam_optimizer.py`：实现了 SAM 优化器。
- `sampro.py`：实现了 SAM 优化器的优化版本。
- `SAMHessian.py`：实现了 SAM Hessian 优化器。

### 训练与评估

- `train_eval.py`：包含训练和评估模型的函数。
- `train_utils.py`：包含训练和评估的辅助函数。
- `train_utils_optimized.py`：包含优化版本的训练和评估辅助函数。
- `train_utils_SAMHessian.py`：包含 SAM Hessian 版本的训练和评估辅助函数。

### 主脚本

- `main.py`：主脚本，负责训练和评估模型。
- `main_optimized.py`：优化版本的主脚本。
- `main_SAMHessian.py`：SAM Hessian 版本的主脚本。

### 结果可视化

- `Grad-CAM.py`：实现了 Grad-CAM，用于可视化模型的注意力区域。

### 日志文件

- `training_log.txt`：记录训练过程的日志文件。
- `training_log_optimized.txt`：记录优化版本的训练过程日志文件。
- `training_log_pro_sam.txt`：记录 SAM 优化器变种的训练过程日志文件。

### 结果存储

- `model/`：存储训练好的模型。
- `model_optimized/`：存储优化版本的训练模型。
- `model_pro_sam/`：存储 SAM 优化器变种的训练模型。
- `image/`：存储训练过程中的图像结果。
- `image_optimized/`：存储优化版本的图像结果。
- `image_pro_sam/`：存储 SAM 优化器变种的图像结果。
- `gradcam_results/`：存储 Grad-CAM 的可视化结果。

## 运行说明

1. 安装依赖：
   ```sh
   pip install -r requirements.txt
   ```
2. 运行主脚本进行训练和评估：
   ```sh
   python main.py
   ```
3. 运行实验不同部分的脚本进行训练和评估（可选）：

   ```sh
   python main_sam.py
   ```

   ```sh
   python main_optimized.py
   ```

   ```sh
   python main_SAMHessian.py
   ```

4. 查看训练日志和结果图像。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进本项目。

## 许可证

本项目采用 MIT 许可证。

## github

项目仓库地址： https://github.com/sowoy153/Course-Design-of-Neural-Networks-and-Deep-Learning.git
