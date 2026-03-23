import torch
import torch.nn.functional as F
import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter
from perlin_numpy import generate_fractal_noise_2d


class DustSimulator:
    def __init__(self, img_size=256):
        self.img_size = img_size

    def generate(self, clean_img, mask, density):
        """
        clean_img: (1,3,H,W) tensor in [0,1]
        mask: (1,1,H,W) tensor (binary panel region)
        density: (1,) tensor in [0,1]
        Returns blueprint: (1,3,H,W) tensor with dust appearance
        """
        batch_size = clean_img.size(0)
        device = clean_img.device
        density = density.item()

        # 1. 生成分形噪声场
        fractal_dim = 2.0 + density * 1.5  # 密度越大分形维数越高
        # 生成多尺度fBM噪声
        noise = torch.zeros(batch_size, 1, self.img_size, self.img_size, device=device)
        for octave in range(1, 5):
            scale = 2 ** octave
            noise_oct = torch.tensor(generate_fractal_noise_2d((self.img_size, self.img_size), octave, lacunarity=2.0,
                                                               persistence=0.5)).float().to(device)
            noise += noise_oct * (0.5 ** octave)
        noise = noise / noise.max()

        # 根据密度阈值二值化
        thresh = 1.0 - density
        binary_mask = (noise > thresh).float()

        # 2. 在HSV空间添加纹理
        # 将clean_img转为HSV
        clean_hsv = self._rgb_to_hsv(clean_img)  # (1,3,H,W)
        # 生成颜色噪声
        color_noise = torch.randn_like(clean_hsv) * 0.1
        # 根据dust类型选择色相偏移
        hue_offset = torch.tensor([0.05, 0.02, 0.01]).view(1, 3, 1, 1).to(device)  # 模拟棕色/灰色
        dust_hsv = clean_hsv + hue_offset * binary_mask
        # 饱和度降低
        dust_hsv[:, 1, :, :] = dust_hsv[:, 1, :, :] * (1 - 0.5 * binary_mask[:, 0, :, :])
        # 亮度降低
        dust_hsv[:, 2, :, :] = dust_hsv[:, 2, :, :] * (1 - 0.3 * binary_mask[:, 0, :, :])
        dust_rgb = self._hsv_to_rgb(dust_hsv)

        # 3. 融合背景：只有panel区域允许灰尘
        dust_rgb = dust_rgb * mask + clean_img * (1 - mask)

        return dust_rgb

    def _rgb_to_hsv(self, rgb):
        # rgb: (N,3,H,W) in [0,1]
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
        maxc = torch.max(rgb, dim=1)[0]
        minc = torch.min(rgb, dim=1)[0]
        delta = maxc - minc
        h = torch.zeros_like(maxc)
        h[delta != 0] = ((g - b) / delta)[delta != 0] % 6
        h = h / 6.0
        s = torch.zeros_like(maxc)
        s[maxc != 0] = (delta / maxc)[maxc != 0]
        v = maxc
        return torch.stack([h, s, v], dim=1)

    def _hsv_to_rgb(self, hsv):
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
        h = h * 6.0
        i = torch.floor(h).long()
        f = h - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        i_mod = i % 6
        r = torch.zeros_like(v)
        g = torch.zeros_like(v)
        b = torch.zeros_like(v)
        r[i_mod == 0] = v[i_mod == 0]
        g[i_mod == 0] = t[i_mod == 0]
        b[i_mod == 0] = p[i_mod == 0]
        r[i_mod == 1] = q[i_mod == 1]
        g[i_mod == 1] = v[i_mod == 1]
        b[i_mod == 1] = p[i_mod == 1]
        r[i_mod == 2] = p[i_mod == 2]
        g[i_mod == 2] = v[i_mod == 2]
        b[i_mod == 2] = t[i_mod == 2]
        r[i_mod == 3] = p[i_mod == 3]
        g[i_mod == 3] = q[i_mod == 3]
        b[i_mod == 3] = v[i_mod == 3]
        r[i_mod == 4] = t[i_mod == 4]
        g[i_mod == 4] = p[i_mod == 4]
        b[i_mod == 4] = v[i_mod == 4]
        r[i_mod == 5] = v[i_mod == 5]
        g[i_mod == 5] = p[i_mod == 5]
        b[i_mod == 5] = q[i_mod == 5]
        return torch.stack([r, g, b], dim=1)