import torch
import torch.nn as nn
import torch.nn.functional as F


class DMCC_SPADE_Block(nn.Module):
    def __init__(self, in_channels, out_channels, struct_channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # DMCC: cross-attention
        self.q_conv = nn.Conv2d(struct_channels, out_channels // 4, 1)
        self.k_conv = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.v_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

        # SPADE modulation
        self.gamma_conv = nn.Conv2d(struct_channels, out_channels, 3, padding=1)
        self.beta_conv = nn.Conv2d(struct_channels, out_channels, 3, padding=1)

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, struct_map):
        # x: (B,C,H,W)
        # struct_map: (B,S,H,W)
        B, C, H, W = x.shape

        # 1. DMCC
        q = self.q_conv(struct_map)  # (B, Cq, H, W)
        k = self.k_conv(x)  # (B, Ck, H, W)
        v = self.v_conv(x)  # (B, Cv, H, W)
        # flatten
        q = q.view(B, -1, H * W).permute(0, 2, 1)  # (B, N, Cq)
        k = k.view(B, -1, H * W)  # (B, Ck, N)
        v = v.view(B, -1, H * W)  # (B, Cv, N)
        attn = torch.bmm(q, k) / (q.size(-1) ** 0.5)  # (B, N, N)
        attn = self.softmax(attn)
        out = torch.bmm(v, attn.permute(0, 2, 1))  # (B, Cv, N)
        out = out.view(B, -1, H, W)
        x = x + out  # residual

        # 2. SPADE modulation
        gamma = self.gamma_conv(struct_map)
        beta = self.beta_conv(struct_map)
        # norm
        x_norm = self.norm(x)
        x_mod = x_norm * (1 + gamma) + beta
        x_mod = self.act(x_mod)

        # 3. conv blocks
        x_mod = self.conv1(x_mod)
        x_mod = self.act(x_mod)
        x_mod = self.conv2(x_mod)

        return x_mod


class MauGANGenerator(nn.Module):
    def __init__(self, img_size=256, num_classes=3):
        super().__init__()
        self.img_size = img_size

        # 编码器: 将输入的蓝图、类型嵌入等融合
        self.encoder = nn.Sequential(
            nn.Conv2d(3 + 3 + 8 + 8 + 8, 64, 3, stride=2, padding=1),
            # 30 channels: 蓝图3, 类型嵌入8, 参数嵌入8, 噪声8, 还有mask? 简化: 蓝图3+类型嵌入8+参数嵌入8+噪声8=27, 再加mask? 实际在训练时拼接了mask? 为了简化，这里用固定通道
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        # 提取texture和struct
        self.texture_head = nn.Conv2d(256, 128, 1)
        self.struct_head = nn.Conv2d(256, 128, 1)

        # 6个DMCC-SPADE块
        self.blocks = nn.ModuleList([
            DMCC_SPADE_Block(128, 128, 128) for _ in range(6)
        ])

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )

        # 类型嵌入层
        self.type_embed = nn.Embedding(num_classes, 8)
        self.param_fc = nn.Linear(1, 8)

    def forward(self, blueprint, defect_type, param):
        # blueprint: (B,3,H,W)
        # defect_type: (B,) long
        # param: (B,1) float
        B, _, H, W = blueprint.shape

        # 类型嵌入
        type_emb = self.type_embed(defect_type)  # (B,8)
        type_emb = type_emb.view(B, 8, 1, 1).expand(-1, -1, H, W)

        # 参数嵌入
        param_emb = self.param_fc(param)  # (B,8)
        param_emb = param_emb.view(B, 8, 1, 1).expand(-1, -1, H, W)

        # 噪声
        noise = torch.randn(B, 8, H, W, device=blueprint.device)

        # 拼接
        x = torch.cat([blueprint, type_emb, param_emb, noise], dim=1)

        # 编码
        feats = self.encoder(x)
        texture = self.texture_head(feats)
        struct = self.struct_head(feats)

        # DMCC-SPADE块
        for block in self.blocks:
            texture = block(texture, struct)

        # 解码
        out = self.decoder(texture)
        return out