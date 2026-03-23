import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from MauGAN.src import MauGANGenerator
from MauGAN.src import MauGANDiscriminator
from MauGAN.src import MauGANLosses
from MauGAN.src import DustSimulator, CrackGenerator, FractureModel
from MauGAN.src import NormalImageDataset


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集
    dataset = NormalImageDataset(args.data_root, img_size=args.img_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 物理模型
    physical_models = {
        'dust': DustSimulator(),
        'crack': CrackGenerator(),
        'glass': FractureModel()
    }
    # 仅启用指定的类型
    active_models = {k: physical_models[k] for k in args.physical_model if k in physical_models}

    # GAN组件
    generator = MauGANGenerator(img_size=args.img_size, num_classes=len(active_models)).to(device)
    discriminator = MauGANDiscriminator(img_size=args.img_size).to(device)

    optimizer_G = optim.AdamW(generator.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999),
                              weight_decay=args.weight_decay)
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=args.decay_step, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=args.decay_step, gamma=0.5)

    losses = MauGANLosses(device)
    writer = SummaryWriter(args.log_dir)

    for epoch in range(args.epochs):
        for i, (clean_img, mask) in enumerate(dataloader):
            clean_img = clean_img.to(device)
            mask = mask.to(device)
            batch_size = clean_img.size(0)

            # 随机选择缺陷类型和参数
            defect_type = torch.randint(0, len(active_models), (batch_size,)).to(device)
            param = torch.rand(batch_size, 1).to(device) * (args.param_range[1] - args.param_range[0]) + \
                    args.param_range[0]

            # 生成物理蓝图
            blueprints = []
            for idx in range(batch_size):
                t = list(active_models.keys())[defect_type[idx].item()]
                model = active_models[t]
                blueprint = model.generate(clean_img[idx:idx + 1], mask[idx:idx + 1], param[idx:idx + 1])
                blueprints.append(blueprint)
            blueprint = torch.cat(blueprints, dim=0)

            # 生成器前向
            fake_img = generator(blueprint, defect_type, param)

            # 鉴别器前向
            d_real = discriminator(clean_img)
            d_fake = discriminator(fake_img.detach())

            # 计算损失
            loss_G, loss_D, loss_dict = losses.compute(
                fake_img, clean_img, blueprint, d_fake, d_real, mask, param
            )

            # 更新生成器
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # 更新鉴别器
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # 记录
            if i % 10 == 0:
                writer.add_scalar('Loss/G', loss_G.item(), epoch * len(dataloader) + i)
                writer.add_scalar('Loss/D', loss_D.item(), epoch * len(dataloader) + i)
                for k, v in loss_dict.items():
                    writer.add_scalar(f'Loss/{k}', v.item(), epoch * len(dataloader) + i)

        scheduler_G.step()
        scheduler_D.step()

        # 保存检查点
        if (epoch + 1) % args.save_every == 0:
            torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, f'generator_epoch{epoch + 1}.pth'))
            torch.save(discriminator.state_dict(),
                       os.path.join(args.checkpoint_dir, f'discriminator_epoch{epoch + 1}.pth'))

    # 最终保存
    torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'generator_final.pth'))
    torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'discriminator_final.pth'))

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--physical_model', nargs='+', default=['dust', 'crack', 'glass'])
    parser.add_argument('--param_range', type=float, nargs=2, default=[0.1, 0.9])
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--decay_step', type=int, default=10000)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_every', type=int, default=5000)
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    main(args)