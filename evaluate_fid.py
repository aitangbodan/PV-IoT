import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models.inception import inception_v3
from scipy.linalg import sqrtm
from MauGAN.src import RealDefectDataset, SynthDefectDataset
from MauGAN.src import compute_activation_statistics


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载真实缺陷数据集（仅用于评估）
    real_dataset = RealDefectDataset(args.real_data_root, img_size=args.img_size)
    real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 加载合成缺陷数据集
    synth_dataset = SynthDefectDataset(args.synth_data_root, split='all', img_size=args.img_size)
    synth_loader = DataLoader(synth_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 加载Inception v3
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()

    # 计算统计量
    mu_real, sigma_real = compute_activation_statistics(real_loader, inception, device)
    mu_synth, sigma_synth = compute_activation_statistics(synth_loader, inception, device)

    # 计算FID
    diff = mu_real - mu_synth
    covmean, _ = sqrtm(sigma_real @ sigma_synth, disp=False)
    fid = diff @ diff + np.trace(sigma_real + sigma_synth - 2 * covmean)
    print(f"FID: {fid:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_data_root', type=str, required=True)
    parser.add_argument('--synth_data_root', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=256)
    args = parser.parse_args()
    main(args)