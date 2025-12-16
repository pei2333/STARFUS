import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================================================
# Minimal UNet with single bottleneck spatial×frequency fusion
# ===============================================================
class DepthwisePointwiseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=in_channels, bias=True
        )
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.act = nn.SiLU()
        self.use_residual = (in_channels == out_channels) and (stride == 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.pw(self.dw(x)))
        return y + x if self.use_residual else y

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = DepthwisePointwiseBlock(in_channels, out_channels, kernel_size=5, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.pre = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.fuse = DepthwisePointwiseBlock(out_channels + skip_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.pre(x)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return x

# per-pixel MLP: 1x1 -> SiLU -> 1x1 
class FrequencyMLP(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.pw1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw2(self.act(self.pw1(x)))

# ---------------------------------------------------------------
# - spatial path: R5 ∘ R5, then 1x1 projection
# - spectral path: per-pixel MLP (1x1 bottleneck)
# - RMSNorm: over spatial dims (h,w)
# - fusion: alpha*(F_spatial ⊙ Fhat_spectral) + beta*(F_spatial + F_spectral)
# - scaled residual connection
# ---------------------------------------------------------------
class SpatialFrequencyFuse(nn.Module):
    def __init__(self, channels: int, hidden: int = 128, res_scale: float = 0.1, eps: float = 1e-5):
        super().__init__()
        # spatial measurement path: two R5 residual blocks
        self.spatial = nn.Sequential(
            DepthwisePointwiseBlock(channels, channels, kernel_size=5, stride=1),
            DepthwisePointwiseBlock(channels, channels, kernel_size=5, stride=1),
        )
        # 1x1 projection after spatial path
        self.s_lin = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        # spectral prior path: per-pixel MLP
        self.freq = FrequencyMLP(channels, hidden)

        # mixing weights
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(0.3))

        # scaled residual
        self.res_scale = nn.Parameter(torch.tensor(res_scale))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # spatial path
        f_s = self.spatial(x)
        f_spatial = self.s_lin(f_s)

        # spectral path
        f_spectral = self.freq(x)

        # RMSNorm over spatial dims (h,w)
        rms = torch.sqrt((f_spectral * f_spectral).mean(dim=(2, 3), keepdim=True) + self.eps)
        fhat_spectral = f_spectral / rms

        # StarFusion
        f_out = self.alpha * (f_spatial * fhat_spectral) + self.beta * (f_spatial + f_spectral)

        return x + self.res_scale * f_out

class StarFusionUNet(nn.Module):
    def __init__(self, out_dim: int = 31, m: int = 256):
        super().__init__()
        in_channels = out_dim + 3

        self.enc1 = DepthwisePointwiseBlock(in_channels, 64, kernel_size=5, stride=1)

        self.down1 = DownBlock(64, 96)
        self.fuse_d1 = SpatialFrequencyFuse(96, hidden=m, res_scale=0.1)

        self.down2 = DownBlock(96, 128)
        self.fuse_d2 = SpatialFrequencyFuse(128, hidden=m, res_scale=0.1)

        self.bottleneck = SpatialFrequencyFuse(128, hidden=m, res_scale=0.1)

        self.up2 = UpBlock(128, skip_channels=96, out_channels=96)
        self.fuse_u2 = SpatialFrequencyFuse(96, hidden=m, res_scale=0.1)

        self.up1 = UpBlock(96, skip_channels=64, out_channels=64)
        self.fuse_u1 = SpatialFrequencyFuse(64, hidden=m, res_scale=0.1)

        self.head = nn.Conv2d(64, out_dim, kernel_size=1, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.head.weight)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)

    def forward(self, LR_up: torch.Tensor, RGB: torch.Tensor) -> torch.Tensor:
        x = torch.cat([LR_up, RGB], dim=1)

        e1 = self.enc1(x)

        e2 = self.down1(e1)
        e2 = self.fuse_d1(e2)

        e3 = self.down2(e2)
        e3 = self.fuse_d2(e3)

        b = self.bottleneck(e3)

        u2 = self.up2(b, e2)
        u2 = self.fuse_u2(u2)

        u1 = self.up1(u2, e1)
        u1 = self.fuse_u1(u1)

        out = self.head(u1)
        return out + LR_up

# ===============================================================
# Models
# ===============================================================
class ModelStarFusionCave(nn.Module):
    def __init__(self, out_dim: int = 31, H: int = 85, W: int = 85, K: int = 4, guide_dim: int = 64, **kwargs):
        super().__init__()
        self.model = StarFusionUNet(out_dim=out_dim, m=kwargs.get('m', 256))

    def forward(self, x: torch.Tensor, up_LR: torch.Tensor, RGB: torch.Tensor) -> torch.Tensor:
        return self.model(up_LR, RGB)

class ModelStarFusionHarvard(nn.Module):
    def __init__(self, out_dim: int = 31, H: int = 85, W: int = 85, K: int = 4, guide_dim: int = 64, **kwargs):
        super().__init__()
        self.model = StarFusionUNet(out_dim=out_dim, m=kwargs.get('m', 256))

    def forward(self, x: torch.Tensor, up_LR: torch.Tensor, RGB: torch.Tensor) -> torch.Tensor:
        return self.model(up_LR, RGB)