import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np


class SynthDefectDataset(Dataset):
    def __init__(self, root, split='train', img_size=640, augment=True):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.augment = augment
        # 假设数据组织：root/images/, root/labels/ (txt格式)
        self.image_dir = os.path.join(root, 'images')
        self.label_dir = os.path.join(root, 'labels')
        self.images = sorted(os.listdir(self.image_dir))
        self.classes = ['dust', 'crack', 'glass']  # 根据实际映射

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        # 读取标签 (YOLO格式: class x_center y_center width height)
        boxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, xc, yc, w, h = map(float, parts)
                    boxes.append([xc, yc, w, h])
                    labels.append(int(cls))
        # 转换为tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        # 对于检测器，需要目标格式
        target = {'boxes': boxes, 'labels': labels}
        return img, target