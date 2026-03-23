import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from MauGAN.src import build_detector


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载检测器
    detector = build_detector('faster_rcnn', num_classes=4, pretrained=False)
    detector.load_state_dict(torch.load(args.checkpoint, map_location=device))
    detector.to(device)
    detector.eval()

    # 预处理
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 读取图像
    img = Image.open(args.image_path).convert('RGB')
    orig_img = np.array(img)
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        outputs = detector(img_tensor)

    # 解析结果
    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()

    # 可视化
    for box, score, label in zip(boxes, scores, labels):
        if score > args.conf_thresh:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(orig_img, f'{label}: {score:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 保存结果
    cv2.imwrite(args.output, orig_img)
    print(f"Detection result saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='detection_result.jpg')
    parser.add_argument('--conf_thresh', type=float, default=0.5)
    args = parser.parse_args()
    main(args)