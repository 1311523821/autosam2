"""
混合引导头模型 (STSF + 轻量解耦 Swin 3D)

结合 STSF 的局部运动检测能力和解耦 Swin 3D 的全局时空建模能力，
用于更准确地定位声呐 B-scan 数据中的小目标。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from operator import mul
from einops import rearrange
from timm.layers import DropPath

# 导入 STSF 编码器
from models.guidance_head import STSFEncoder, ConvBlock


# ============================================================
# 从 LVNet 移植的核心组件
# ============================================================

class Mlp(nn.Module):
    """多层感知机"""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Conv3DMlp(nn.Module):
    """带有 3D 深度可分离卷积的 MLP，强制局部时空平滑"""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)

        # 3x3x3 深度可分离卷积
        self.dwconv3d = nn.Conv3d(
            hidden_features, hidden_features,
            kernel_size=(3, 3, 3), padding=(1, 1, 1), groups=hidden_features
        )

        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: (B, D, H, W, C)
        B, D, H, W, C = x.shape
        x = self.fc1(x)

        # 转置为 Conv3d 需要的形状 (B, C, D, H, W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.dwconv3d(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    将输入划分为窗口
    Args:
        x: (B, D, H, W, C)
        window_size: (wD, wH, wW)
    Returns:
        windows: (B*num_windows, window_size[0]*window_size[1]*window_size[2], C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0],
               H // window_size[1], window_size[1],
               W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    窗口逆操作
    Args:
        windows: (B*num_windows, window_size_prod, C)
        window_size: (wD, wH, wW)
    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2],
                     window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """计算实际使用的窗口大小"""
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0
    if shift_size is not None:
        return tuple(use_window_size), tuple(use_shift_size)
    return tuple(use_window_size)


class TemporalAxialAttention(nn.Module):
    """
    纯时序轴向注意力 (沿 D 轴)
    输入: (B, D, H, W, C) -> 重整为 (B*H*W, D, C) 计算序列自注意力
    返回: (B, D, H, W, C)
    """
    def __init__(self, dim, num_heads, qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        B, D, H, W, C = x.shape
        # 将所有空间位置压入 batch 维，序列长度为 D
        x = rearrange(x, 'b d h w c -> (b h w) d c')
        N, L, C = x.shape  # N = B*H*W, L = D

        # 多头自注意力 (沿 D 维)
        qkv = self.qkv(x).reshape(N, L, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(N, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # 恢复原始形状
        x = rearrange(x, '(b h w) d c -> b d h w c', b=B, h=H, w=W)
        return x


class WindowAttention3D(nn.Module):
    """
    3D 窗口注意力 (空间窗口，跨帧)
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (wD, wH, wW)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads)
        )

        # 相对位置索引
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows*B, window_size_prod, C)
            mask: (num_windows, window_size_prod, window_size_prod) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ============================================================
# 轻量化解耦 Swin 3D Block
# ============================================================

class LiteDecoupledSwinBlock(nn.Module):
    """
    轻量化解耦时空 Swin Transformer Block

    相比原版:
    1. 只保留时序轴向注意力（核心组件）
    2. 可选的空间注意力
    3. 简化的 MLP
    """
    def __init__(self, dim, num_heads=2, window_size=(2, 7, 7),
                 mlp_ratio=2., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., use_spatial_attn=False, use_conv3d_mlp=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.use_spatial_attn = use_spatial_attn

        # 时序轴向注意力（核心组件）
        self.norm_temporal = nn.LayerNorm(dim)
        self.temporal_attn = TemporalAxialAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            drop=drop, attn_drop=attn_drop
        )

        # 可选的空间注意力
        if use_spatial_attn:
            self.norm_spatial = nn.LayerNorm(dim)
            self.spatial_attn = WindowAttention3D(
                dim, window_size=window_size, num_heads=num_heads,
                qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
            )

        # MLP
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if use_conv3d_mlp:
            self.mlp = Conv3DMlp(
                in_features=dim, hidden_features=mlp_hidden_dim, drop=drop
            )
        else:
            self.mlp = Mlp(
                in_features=dim, hidden_features=mlp_hidden_dim, drop=drop
            )

    def forward(self, x):
        """
        Args:
            x: (B, D, H, W, C)
        Returns:
            x: (B, D, H, W, C)
        """
        # 可选的空间注意力
        if self.use_spatial_attn:
            # 简化的空间注意力（不使用 mask_matrix）
            shortcut = x
            x = self.norm_spatial(x)
            B, D, H, W, C = x.shape

            # 简单的窗口划分
            window_size = get_window_size((D, H, W), self.window_size)
            x_windows = window_partition(x, window_size)
            attn_windows = self.spatial_attn(x_windows, mask=None)
            attn_windows = attn_windows.view(-1, *window_size, C)
            x = window_reverse(attn_windows, window_size, B, D, H, W)
            x = shortcut + self.drop_path(x)
        # 时序轴向注意力
        shortcut = x
        x = self.norm_temporal(x)
        x = self.temporal_attn(x)
        x = shortcut + self.drop_path(x)


        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ============================================================
# 混合引导头模型
# ============================================================

class GuidanceHeadSwin(nn.Module):
    """
    STSF + 解耦 Swin 3D 混合引导头

    架构:
    1. STSF Encoder: 局部运动检测（初筛），输出保留时序维度
    2. Lite Decoupled Swin 3D: 全局时空注意力抑噪（精筛）
    3. MLP Head: 热力图定位
    """
    def __init__(
        self,
        in_chans: int = 1,
        embed_dim: int = 64,
        hidden_dim: int = 32,
        num_frame: int = 4,
        swin_depth: int = 1,
        swin_heads: int = 2,
        swin_window_size: tuple = (2, 7, 7),
        use_spatial_attn: bool = False,
        use_conv3d_mlp: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frame = num_frame

        # 1. STSF 编码器 - 保留时序维度
        self.encoder = STSFEncoder(in_chans, embed_dim, num_frame, keep_temporal=True)

        # 2. 轻量化解耦 Swin 3D
        self.swin_blocks = nn.ModuleList([
            LiteDecoupledSwinBlock(
                dim=embed_dim,
                num_heads=swin_heads,
                window_size=swin_window_size,
                mlp_ratio=2.,
                use_spatial_attn=use_spatial_attn,
                use_conv3d_mlp=use_conv3d_mlp
            )
            for _ in range(swin_depth)
        ])

        # 3. 时间池化 + 解码器
        self.decoder = nn.Sequential(
            ConvBlock(embed_dim, hidden_dim, 3, padding=1),
            ConvBlock(hidden_dim, hidden_dim, 3, padding=1),
            nn.Conv2d(hidden_dim, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, D, H, W) - D 帧灰度图
        Returns:
            heatmap: (B, 1, H, W) - 热力图 logits
        """
        B, C, D, H, W = x.shape

        # 1. STSF 编码 - 现在输出 (B, C, D, H, W)，保留时序维度
        features = self.encoder(x)  # (B, C, D, H, W)

        # 重塑为 (B, D, H, W, C) 用于 Swin
        features = features.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)

        # 2. 解耦 Swin 3D 处理 - 现在处理的是真正的时序序列
        for swin_block in self.swin_blocks:
            features = swin_block(features)

        # 3. 时间池化
        features = features.mean(dim=1)  # (B, H, W, C)
        features = features.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # 4. 解码输出
        heatmap = self.decoder(features)  # (B, 1, H, W)

        return heatmap

    def get_prompt_point(self, heatmap: torch.Tensor, threshold: float = None):
        """
        从热力图提取最高置信度点
        Args:
            heatmap: (B, 1, H, W) logits
            threshold: 可选阈值
        Returns:
            points: (B, 2) - (x, y) 坐标
            confidences: (B,) - 置信度
        """
        pred = torch.sigmoid(heatmap)
        B, _, H, W = pred.shape
        pred_flat = pred.view(B, -1)

        max_vals, max_idx = torch.max(pred_flat, dim=1)
        y = max_idx // W
        x = max_idx % W

        return torch.stack([x, y], dim=1).float(), max_vals


def build_guidance_head_swin(
    embed_dim: int = 64,
    hidden_dim: int = 32,
    swin_depth: int = 1,
    swin_heads: int = 2,
    use_spatial_attn: bool = False
) -> GuidanceHeadSwin:
    """构建混合引导头模型"""
    return GuidanceHeadSwin(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        swin_depth=swin_depth,
        swin_heads=swin_heads,
        use_spatial_attn=use_spatial_attn
    )


if __name__ == '__main__':
    # 测试模型
    model = GuidanceHeadSwin(
        embed_dim=64,
        hidden_dim=32,
        swin_depth=1,
        swin_heads=2
    )

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 测试前向传播
    x = torch.randn(2, 1, 4, 256, 256)
    with torch.no_grad():
        output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 测试提取点
    points, confs = model.get_prompt_point(output)
    print(f"预测点: {points}")
    print(f"置信度: {confs}")
