import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _make_layer(block, num_blocks, **kwargs):
    return nn.Sequential(*(block(**kwargs) for _ in range(num_blocks)))


class StateSpaceGlobalBlock(nn.Module):
    def __init__(self, channels: int, d_state: int = 16):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:
            raise ImportError(
                "MSGRDBNet with global_block_type='ssgm' requires mamba-ssm. "
                "Install it with `pip install mamba-ssm`."
            ) from exc

        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=4,
            expand=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.mamba(self.norm(x))
        return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class WindowAttentionGlobalBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        window_size: int = 8,
        num_heads: int = 4,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        self.channels = channels
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def _partition(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        ws = self.window_size
        x = x.view(b, h // ws, ws, w // ws, ws, c)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, c)

    def _reverse(self, windows: torch.Tensor, b: int, h: int, w: int) -> torch.Tensor:
        ws = self.window_size
        c = self.channels
        x = windows.view(b, h // ws, w // ws, ws, ws, c)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ws = self.window_size
        x = x.permute(0, 2, 3, 1).contiguous()

        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h), mode="reflect")

        _, hp, wp, _ = x.shape
        windows = self._partition(x).view(-1, ws * ws, c)
        windows = self.norm(windows)

        qkv = self.qkv(windows).view(-1, ws * ws, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(-1, ws * ws, c)
        out = self.proj(out).view(-1, ws, ws, c)
        out = self._reverse(out, b, hp, wp)
        out = out[:, :h, :w, :]
        return out.permute(0, 3, 1, 2).contiguous()


class MSGCSSABlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes=(3, 5, 7),
        reduction: int = 8,
        enable_scale_attn: bool = False,
        global_block_type: str = "ssgm",
        global_res_scale: float = 0.1,
        window_size: int = 8,
        num_heads: int = 4,
        d_state: int = 16,
    ):
        super().__init__()
        if global_block_type not in {"ssgm", "window_attn", "none"}:
            raise ValueError("global_block_type must be: ssgm, window_attn, or none")

        self.num_scales = len(kernel_sizes)
        self.enable_scale_attn = enable_scale_attn
        self.global_block_type = global_block_type
        self.global_res_scale = global_res_scale

        self.ms_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, k, 1, k // 2),
                nn.LeakyReLU(0.2, inplace=True),
            )
            for k in kernel_sizes
        )

        global_channels = out_channels * self.num_scales
        hidden_channels = max(out_channels // reduction, 4)
        self.scale_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(global_channels, hidden_channels, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, self.num_scales, 1),
            nn.Softmax(dim=1),
        )

        if global_block_type == "ssgm":
            self.global_block = StateSpaceGlobalBlock(global_channels, d_state=d_state)
        elif global_block_type == "window_attn":
            self.global_block = WindowAttentionGlobalBlock(
                global_channels, window_size=window_size, num_heads=num_heads
            )
        else:
            self.global_block = nn.Identity()

        self.fusion = nn.Conv2d(global_channels, out_channels, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [conv(x) for conv in self.ms_convs]
        local_feat = torch.cat(feats, dim=1)

        if self.enable_scale_attn:
            weights = self.scale_attn(local_feat)
            weights = weights.unsqueeze(2).expand(-1, -1, feats[0].shape[1], -1, -1)
            local_feat = local_feat * weights.reshape(x.shape[0], -1, 1, 1)
        else:
            local_feat = local_feat / self.num_scales

        if self.global_block_type != "none":
            local_feat = local_feat + self.global_res_scale * self.global_block(local_feat)

        return self.lrelu(self.fusion(local_feat))


class MSGRDB(nn.Module):
    def __init__(
        self,
        num_feat: int = 64,
        num_grow_ch: int = 32,
        kernel_sizes=(3, 5, 7),
        enable_scale_attn: bool = False,
        global_block_type: str = "ssgm",
        global_res_scale: float = 0.1,
        window_size: int = 8,
        num_heads: int = 4,
        d_state: int = 16,
    ):
        super().__init__()
        common = dict(
            out_channels=num_grow_ch,
            kernel_sizes=kernel_sizes,
            enable_scale_attn=enable_scale_attn,
            global_block_type=global_block_type,
            global_res_scale=global_res_scale,
            window_size=window_size,
            num_heads=num_heads,
            d_state=d_state,
        )
        self.b1 = MSGCSSABlock(num_feat, **common)
        self.b2 = MSGCSSABlock(num_feat + num_grow_ch, **common)
        self.b3 = MSGCSSABlock(num_feat + 2 * num_grow_ch, **common)
        self.b4 = MSGCSSABlock(num_feat + 3 * num_grow_ch, **common)
        self.conv_fuse = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.b1(x)
        x2 = self.b2(torch.cat((x, x1), dim=1))
        x3 = self.b3(torch.cat((x, x1, x2), dim=1))
        x4 = self.b4(torch.cat((x, x1, x2, x3), dim=1))
        residual = self.conv_fuse(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x + 0.2 * residual


class MSGRDBNet(nn.Module):
    def __init__(
        self,
        scale: int = 4,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 50,
        num_grow_ch: int = 32,
        kernel_sizes=(3, 5, 7),
        enable_scale_attn: bool = False,
        global_block_type: str = "ssgm",
        global_res_scale: float = 0.1,
        window_size: int = 8,
        num_heads: int = 4,
        d_state: int = 16,
    ):
        super().__init__()
        if scale < 1 or scale & (scale - 1):
            raise ValueError("scale must be a positive power of 2")

        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = _make_layer(
            MSGRDB,
            num_block,
            num_feat=num_feat,
            num_grow_ch=num_grow_ch,
            kernel_sizes=tuple(kernel_sizes),
            enable_scale_attn=enable_scale_attn,
            global_block_type=global_block_type,
            global_res_scale=global_res_scale,
            window_size=window_size,
            num_heads=num_heads,
            d_state=d_state,
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.num_upsample = int(math.log2(scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        for _ in range(self.num_upsample):
            feat = F.interpolate(feat, scale_factor=2, mode="nearest")
            feat = self.lrelu(self.conv_up(feat))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))
