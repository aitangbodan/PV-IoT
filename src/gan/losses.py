import torch
import torch.nn as nn
import torch.nn.functional as F


class MauGANLosses:
    def __init__(self, device, lambda_adv=1.0, lambda_fm=10.0, lambda_struct=5.0, lambda_bg=10.0):
        self.device = device
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.lambda_struct = lambda_struct
        self.lambda_bg = lambda_bg
        self.criterion_gan = nn.MSELoss()  # LSGAN
        self.criterion_l1 = nn.L1Loss()

    def compute(self, fake_img, real_img, blueprint, d_fake, d_real, mask, param):
        # 对抗损失
        real_labels = torch.ones_like(d_real).to(self.device)
        fake_labels = torch.zeros_like(d_fake).to(self.device)
        loss_adv_G = self.criterion_gan(d_fake, real_labels)
        loss_adv_D = (self.criterion_gan(d_real, real_labels) + self.criterion_gan(d_fake, fake_labels)) * 0.5

        # 特征匹配损失（简化：使用鉴别器中间层？这里用L1）
        loss_fm = self.criterion_l1(fake_img, real_img)

        # 结构一致性损失（重建蓝图）
        # 需要额外结构鉴别器输出，这里我们用一个预训练或简单的CNN，但为了简化，我们用L1在蓝图和fake_img之间？
        # 实际上需要将fake_img输入结构鉴别器，但为简化，我们假设结构鉴别器输出与blueprint的L1
        # 这里我们调用一个简化的结构鉴别器（在discriminator里已实现）
        # 我们假设在训练过程中已经得到了struct_out_fake
        # 为了代码完整，我们在compute时重新计算结构损失
        # 注意：实际训练时discriminator forward会返回两个值，这里我们传入struct_fake, struct_real
        # 但为了简化，我们假设外部已经计算好并作为参数传入。这里我们只计算蓝图和生成的图像之间的L1作为替代
        loss_struct = self.criterion_l1(fake_img, blueprint)

        # 背景保持损失
        # mask为1表示缺陷区域，0表示背景
        bg_mask = 1 - mask
        loss_bg = self.criterion_l1(fake_img * bg_mask, real_img * bg_mask)

        loss_G = self.lambda_adv * loss_adv_G + self.lambda_fm * loss_fm + self.lambda_struct * loss_struct + self.lambda_bg * loss_bg
        loss_D = loss_adv_D

        loss_dict = {
            'adv_G': loss_adv_G,
            'fm': loss_fm,
            'struct': loss_struct,
            'bg': loss_bg,
            'adv_D': loss_adv_D
        }

        return loss_G, loss_D, loss_dict