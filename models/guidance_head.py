"""
引导头模型 (Guidance Head Model) - 增强版

增强模型容量，添加更多特征提取层和注意力机制。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """基础卷积块"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class STSFEncoder(nn.Module):
    """
    时空静态滤波编码器 (增强版)

    增加了更多通道和残差连接。

    参数:
        keep_temporal: 是否保留时序维度。True 时输出 (B, C, D, H, W)，False 时输出 (B, C, H, W)
    """

    def __init__(self, in_chans: int = 1, out_chans: int = 64, num_frame: int = 4, keep_temporal: bool = False):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_frame = num_frame
        self.keep_temporal = keep_temporal

        # 初始特征提取
        self.init_conv = nn.Sequential(
            ConvBlock(in_chans, 32, 3, padding=1),
            ConvBlock(32, 32, 3, padding=1),
        )

        # 多尺度空间卷积（全部使用padding=kernel_size//2保持尺寸）
        self.conv1 = ConvBlock(32, out_chans, 1, padding=0)
        self.conv2 = ConvBlock(32, out_chans, 3, padding=1)
        self.conv3 = ConvBlock(32, out_chans, 5, padding=2)
        self.conv4 = ConvBlock(32, out_chans, 7, padding=3)

        # 时序方差分支
        self.var_conv = nn.Sequential(
            ConvBlock(in_chans, 32, 3, padding=1),
            ConvBlock(32, out_chans, 3, padding=1),
        )

        # 融合层
        self.fusion = nn.Sequential(
            ConvBlock(out_chans * 4, out_chans * 2, 1, padding=0),
            ConvBlock(out_chans * 2, out_chans, 1, padding=0),
        )

        # 残差投影
        self.residual = ConvBlock(32, out_chans, 1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, D, H, W) 输入帧序列
        Returns:
            如果 keep_temporal=True: (B, C, D, H, W) 保留时序维度
            如果 keep_temporal=False: (B, C, H, W) 时序聚合结果
        """
        if x.dim() == 5:
            B, C, D, H, W = x.shape
            x_flat = x.view(B * D, C, H, W)
        else:
            x_flat = x
            D = self.num_frame
            B = x_flat.shape[0] // D

        bd, c, h, w = x_flat.shape

        # 初始特征
        feat = self.init_conv(x_flat)

        # 多尺度空间特征
        x1 = self.conv1(feat)
        x2 = self.conv2(feat)
        x3 = self.conv3(feat)
        x4 = self.conv4(feat)

        # 时序方差特征
        x_3d = x_flat.view(B, D, c, h, w)
        temporal_var = torch.var(x_3d, dim=1, keepdim=True, unbiased=False)
        temporal_var = temporal_var.expand(-1, D, -1, -1, -1)
        var_flat = temporal_var.reshape(bd, c, h, w)
        x_var = self.var_conv(var_flat)

        # 融合（只用空间特征，var用于后续）
        multi_scale = torch.cat([x1, x2, x3, x4], dim=1)
        fused = self.fusion(multi_scale)

        # 残差
        res = self.residual(feat)

        # 输出 + 方差特征
        x_out = fused + res + x_var

        # 重塑为时序形式
        x_out = x_out.view(B, D, self.out_chans, h, w)  # (B, D, C, H, W)

        if self.keep_temporal:
            # 保留时序维度，输出 (B, C, D, H, W)
            x_out = x_out.permute(0, 2, 1, 3, 4).contiguous()
        else:
            # 时序聚合，输出 (B, C, H, W)
            x_out = x_out.mean(dim=1)

        return x_out


class LightweightDecoder(nn.Module):
    """增强解码器"""

    def __init__(self, in_chans: int = 64, hidden_chans: int = 32):
        super().__init__()
        self.conv1 = ConvBlock(in_chans, hidden_chans, 3, padding=1)
        self.conv2 = ConvBlock(hidden_chans, hidden_chans, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_chans, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class GuidanceHead(nn.Module):
    """引导头模型（增强版）"""

    def __init__(
        self,
        in_chans: int = 1,
        embed_dim: int = 64,
        hidden_dim: int = 32,
        num_frame: int = 4
    ):
        super().__init__()
        self.encoder = STSFEncoder(in_chans, embed_dim, num_frame)
        self.decoder = LightweightDecoder(embed_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        heatmap = self.decoder(features)
        return heatmap

    def get_prompt_point(
        self,
        heatmap: torch.Tensor,
        threshold: Optional[float] = None
    ) -> torch.Tensor:
        if heatmap.dim() == 4:
            B, C, H, W = heatmap.shape
        else:
            raise ValueError(f"期望4D张量，得到{heatmap.dim()}D")

        heatmap_prob = torch.sigmoid(heatmap)

        if threshold is not None:
            heatmap_prob = heatmap_prob * (heatmap_prob > threshold).float()

        heatmap_flat = heatmap_prob.view(B, -1)
        max_idx = torch.argmax(heatmap_flat, dim=1)

        y = max_idx // W
        x = max_idx % W

        return torch.stack([x, y], dim=1).float()


def build_guidance_head(config: dict) -> GuidanceHead:
    return GuidanceHead(
        in_chans=config.get('in_chans', 1),
        embed_dim=config.get('embed_dim', 64),
        hidden_dim=config.get('hidden_dim', 32),
        num_frame=config.get('num_frame', 4)
    )


if __name__ == '__main__':
    model = GuidanceHead(in_chans=1, embed_dim=64, hidden_dim=32, num_frame=4)
    x = torch.randn(2, 1, 4, 256, 256)

    heatmap = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {heatmap.shape}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    points = model.get_prompt_point(heatmap)
    print(f"Prompt points shape: {points.shape}")
