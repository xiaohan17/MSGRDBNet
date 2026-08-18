import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _covariance(sig_x, sig_y, radians):
    sig_x = sig_x.view(-1, 1, 1)
    sig_y = sig_y.view(-1, 1, 1)
    radians = radians.view(-1, 1, 1)
    zeros = torch.zeros_like(sig_x)
    d = torch.cat((torch.cat((sig_x.square(), zeros), 2), torch.cat((zeros, sig_y.square()), 2)), 1)
    u = torch.cat(
        (
            torch.cat((radians.cos(), -radians.sin()), 2),
            torch.cat((radians.sin(), radians.cos()), 2),
        ),
        1,
    )
    return torch.bmm(u, torch.bmm(d, u.transpose(1, 2)))


def _gaussian_kernel(batch, kernel_size, device, blur_type, randomize, cfg):
    axis = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
    xx = axis.repeat(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    yy = axis.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)

    if blur_type == "iso_gaussian":
        if randomize:
            sigma = torch.rand(batch, device=device) * (cfg["sig_max"] - cfg["sig_min"]) + cfg["sig_min"]
        else:
            sigma = torch.full((batch,), float(cfg["sig"]), device=device)
        kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma.view(-1, 1, 1).square()))
    elif blur_type == "aniso_gaussian":
        if randomize:
            lambda_1 = torch.rand(batch, device=device) * (cfg["lambda_max"] - cfg["lambda_min"]) + cfg["lambda_min"]
            lambda_2 = torch.rand(batch, device=device) * (cfg["lambda_max"] - cfg["lambda_min"]) + cfg["lambda_min"]
            theta = torch.rand(batch, device=device) * math.pi
        else:
            lambda_1 = torch.full((batch,), float(cfg["lambda_1"]), device=device)
            lambda_2 = torch.full((batch,), float(cfg["lambda_2"]), device=device)
            theta = torch.full((batch,), float(cfg.get("theta", 0.0)) * math.pi / 180.0, device=device)
        covariance = _covariance(lambda_1, lambda_2, theta)
        xy = torch.stack((xx, yy), -1).view(batch, -1, 2)
        inv = torch.inverse(covariance)
        kernel = torch.exp(-0.5 * (torch.bmm(xy, inv) * xy).sum(2)).view(batch, kernel_size, kernel_size)
    else:
        raise ValueError("blur_type must be iso_gaussian or aniso_gaussian")

    return kernel / kernel.sum((1, 2), keepdim=True)


class MatlabBicubic(nn.Module):
    def _cubic(self, x):
        ax = x.abs()
        ax2 = ax.square()
        ax3 = ax2 * ax
        c1 = (ax <= 1).float()
        c2 = ((ax > 1) & (ax <= 2)).float()
        return (1.5 * ax3 - 2.5 * ax2 + 1) * c1 + (-0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2) * c2

    def _contribute(self, in_size, out_size, scale, device):
        kernel_width = 4 / scale if scale < 1 else 4
        x = torch.arange(1, out_size + 1, device=device, dtype=torch.float32)
        u = x / scale + 0.5 * (1 - 1 / scale)
        left = torch.floor(u - kernel_width / 2)
        p = int(np.ceil(kernel_width)) + 2
        indices = left[:, None] + torch.arange(p, device=device, dtype=torch.float32)[None, :]
        distance = u[:, None] - indices
        weights = scale * self._cubic(distance * scale) if scale < 1 else self._cubic(distance)
        weights = weights / weights.sum(1, keepdim=True)
        indices = indices.clamp(1, in_size).long() - 1
        keep = weights.abs().sum(0) > 0
        return weights[:, keep], indices[:, keep]

    def forward(self, x, scale):
        _, _, h, w = x.shape
        out_h, out_w = int(h * scale), int(w * scale)
        weight_h, index_h = self._contribute(h, out_h, scale, x.device)
        weight_w, index_w = self._contribute(w, out_w, scale, x.device)

        temp = x[:, :, index_h, :] * weight_h[None, None, :, :, None]
        temp = temp.sum(3).permute(0, 1, 3, 2)
        out = temp[:, :, index_w, :] * weight_w[None, None, :, :, None]
        return out.sum(3).permute(0, 1, 3, 2)


class SRMDDegradation(nn.Module):
    def __init__(self, scale=4, **cfg):
        super().__init__()
        self.scale = int(scale)
        self.kernel_size = int(cfg.get("blur_kernel", 21))
        self.blur_type = cfg.get("blur_type", "iso_gaussian")
        self.noise = float(cfg.get("noise", 0.0))
        self.cfg = {
            "sig": float(cfg.get("sig", 2.6)),
            "sig_min": float(cfg.get("sig_min", 0.2)),
            "sig_max": float(cfg.get("sig_max", 4.0)),
            "lambda_1": float(cfg.get("lambda_1", 0.2)),
            "lambda_2": float(cfg.get("lambda_2", 4.0)),
            "lambda_min": float(cfg.get("lambda_min", 0.2)),
            "lambda_max": float(cfg.get("lambda_max", 4.0)),
            "theta": float(cfg.get("theta", 0.0)),
        }
        self.downsample = MatlabBicubic()

    def _blur(self, x, kernels):
        b, c, h, w = x.shape
        pad = self.kernel_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        x = x.view(1, b * c, x.shape[-2], x.shape[-1])
        kernels = kernels[:, None].repeat(1, c, 1, 1).view(b * c, 1, self.kernel_size, self.kernel_size)
        return F.conv2d(x, kernels, groups=b * c).view(b, c, h, w)

    @torch.no_grad()
    def forward(self, hr, randomize=True):
        x = hr.clamp(0, 1) * 255.0
        kernels = _gaussian_kernel(
            x.shape[0], self.kernel_size, x.device, self.blur_type, randomize, self.cfg
        )
        x = self._blur(x, kernels)
        x = self.downsample(x, 1.0 / self.scale)

        if self.noise > 0:
            if randomize:
                level = torch.rand(x.shape[0], 1, 1, 1, device=x.device) * self.noise
            else:
                level = torch.full((x.shape[0], 1, 1, 1), self.noise, device=x.device)
            x = x + torch.randn_like(x) * level

        return x.round().clamp(0, 255) / 255.0
