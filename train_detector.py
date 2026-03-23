import argparse
import os
import torch
from torch.utils.data import DataLoader
from MauGAN.src import SynthDefectDataset
from MauGAN.src import DetectorTrainer
from MauGAN.src import build_detector


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建数据集
    train_dataset = SynthDefectDataset(
        args.syn_data_root,
        split='train',
        img_size=args.img_size,
        augment=True
    )
    val_dataset = SynthDefectDataset(
        args.syn_data_root,
        split='val',
        img_size=args.img_size,
        augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                            collate_fn=lambda x: tuple(zip(*x)))

    # 构建检测器
    detector = build_detector(args.detector, num_classes=len(train_dataset.classes), pretrained=args.pretrained)
    detector.to(device)

    # 训练器
    trainer = DetectorTrainer(
        detector,
        device,
        lr=args.lr,
        epochs=args.epochs,
        save_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    trainer.train(train_loader, val_loader)

    # 保存最终模型
    torch.save(detector.state_dict(), os.path.join(args.checkpoint_dir, 'detector_final.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--syn_data_root', type=str, required=True)
    parser.add_argument('--detector', type=str, default='faster_rcnn', choices=['faster_rcnn', 'yolo', 'retinanet'])
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs_detector')
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    main(args)