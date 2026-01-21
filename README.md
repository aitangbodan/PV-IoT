# MauGAN: Physics-Constrained Generative Modeling for Photovoltaic Defect Detection

**MauGAN** is the official PyTorch implementation for the paper: **“MauGAN: Physics-Constrained Generative Modeling for Photovoltaic Defect Detection with Zero Real Defect Annotations”**.

This repository contains the source code and models for our novel two-stage framework that synthesizes **physically plausible and visually realistic** defect images for industrial visual inspection, enabling high-performance detection models to be trained **without any real defect annotations**.

## 📦 安装

1.  **克隆仓库**
    ```bash
    git clone https://github.com/aitangbodan/PV-IoT
    cd maugan
    ```

2.  **创建并激活Conda环境（推荐）**
    ```bash
    conda create -n maugan python=3.8
    conda activate maugan
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```
    *主要依赖：PyTorch, torchvision, numpy, opencv-python, scikit-image, tensorboard*

## 🚀 快速开始

### 1. 数据准备
请将您的正常（无缺陷）光伏面板图像放入 `data/train/clean/` 目录下。我们提供了一个示例数据集结构供参考。

### 2. 训练MauGAN生成器
执行以下命令开始训练物理引导的缺陷生成模型：
```bash
python train_generator.py \
    --data_root ./data/train \
    --physical_model dust crack  # 指定要合成的缺陷类型
    --param_range 0.1 0.9       # 设置物理参数（如密度）的随机范围
    --output_dir ./results/generated_defects
```

### 3. 使用合成数据训练检测器
利用生成的缺陷图像和对应的边界框标注（自动生成）训练一个检测器：
```bash
python train_detector.py \
    --syn_data_root ./results/generated_defects \
    --detector faster_rcnn      # 可选: faster_rcnn, yolo, retinaNet
    --epochs 50
```

### 4. 在您自己的图像上推理
使用训练好的检测模型对新图像进行缺陷检测：
```bash
python inference.py \
    --image_path /path/to/your/test_image.jpg \
    --checkpoint ./checkpoints/detector_best.pth \
    --output ./detection_result.jpg
```

## 📂 项目结构

```
maugan/
├── README.md
├── requirements.txt
├── configs/                 # 配置文件
├── data/                    # 数据目录
├── src/
│   ├── physical_models/     # 核心：物理缺陷生成模块（灰尘、裂纹、破碎）
│   │   ├── dust_simulator.py
│   │   ├── crack_generator.py
│   │   └── fracture_model.py
│   ├── gan/                 # 多模态对抗增强网络
│   │   ├── generator.py
│   │   ├── discriminator.py
│   │   └── losses.py
│   ├── detection/           # 下游检测器适配与训练
│   │   ├── dataset_synth.py
│   │   └── trainer.py
│   └── utils/               # 工具函数（可视化、指标计算）
├── scripts/                 # 训练和评估脚本
├── checkpoints/             # 预训练模型（论文结果复现）
└── examples/                # 输入/输出示例图
```

## 📈 实验结果复现

我们提供了在论文中报告的关键结果的复现脚本和预训练模型：
- **生成质量**：运行 `evaluate_fid.py` 计算生成图像的FID分数（目标<20）。
- **检测性能**：使用 `checkpoints/` 下的预训练检测器，在提供的验证集上运行评估，预期可达到论文中报告的F1分数提升。

## 📝 引用

如果MauGAN对您的研究有帮助，请引用我们的论文：

```bibtex
@article{wang2026maugan,
  title={MauGAN: Physics-Constrained Generative Modeling for Photovoltaic Defect Detection with Zero Real Defect Annotations},
  author={Wang, X. and et al.},
  journal={Submitted to Computer Vision and Image Understanding},
  year={2026}
}
```

## 📄 许可证

本项目采用 **MIT 许可证**。详情请见 [LICENSE](LICENSE) 文件。
