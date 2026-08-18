import math

import torch
import torch.nn.functional as F


def _crop(x, border):
    if border <= 0:
        return x
    return x[..., border:-border, border:-border]


def psnr(sr, hr, crop_border=0):
    sr = _crop(sr.float(), crop_border)
    hr = _crop(hr.float(), crop_border)
    mse = F.mse_loss(sr, hr).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _gaussian_window(size=11, sigma=1.5, device="cpu", dtype=torch.float32):
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords.square()) / (2 * sigma * sigma))
    g = g / g.sum()
    return torch.outer(g, g)


def ssim(sr, hr, crop_border=0):
    sr = _crop(sr.float(), crop_border)
    hr = _crop(hr.float(), crop_border)
    c = sr.shape[1]
    window = _gaussian_window(device=sr.device, dtype=sr.dtype)[None, None].repeat(c, 1, 1, 1)
    mu_x = F.conv2d(sr, window, padding=5, groups=c)
    mu_y = F.conv2d(hr, window, padding=5, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x2 = F.conv2d(sr * sr, window, padding=5, groups=c) - mu_x2
    sigma_y2 = F.conv2d(hr * hr, window, padding=5, groups=c) - mu_y2
    sigma_xy = F.conv2d(sr * hr, window, padding=5, groups=c) - mu_xy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return score.mean().item()
