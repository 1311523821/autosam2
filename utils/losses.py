"""
损失函数 (Loss Functions)

用于引导头训练的损失函数，特别针对小目标热力图预测优化。

核心组件：
- HeatmapFocalLoss: CenterNet风格的热力图损失，解决正负样本不平衡问题
- MSEHeatmapLoss: 标准MSE损失（用于对比实验）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HeatmapFocalLoss(nn.Module):
    """
    热力图损失函数 - 使用加权BCE

    核心思想：对小目标，需要极端的正样本权重才能平衡梯度。
    alpha 和 beta 参数保留用于未来的Focal Loss实现，当前使用加权BCE。
    """

    def __init__(self, alpha: int = 2, beta: int = 4, pos_weight: float = 500.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.pos_weight = pos_weight

    def forward(
        self,
        pred_logits: torch.Tensor,
        gt_heatmap: torch.Tensor
    ) -> torch.Tensor:
        """
        计算加权BCE损失

        Args:
            pred_logits: 网络输出，未经过sigmoid
            gt_heatmap: 高斯热力图标签

        Returns:
            loss: 标量损失
        """
        if pred_logits.dim() == 3:
            pred_logits = pred_logits.unsqueeze(1)
        if gt_heatmap.dim() == 4:
            gt_heatmap = gt_heatmap.squeeze(1)

        # 使用BCEWithLogits（autocast安全）
        bce = F.binary_cross_entropy_with_logits(
            pred_logits.squeeze(1), gt_heatmap, reduction='none'
        )

        # 权重：正样本区域获得极高权重
        # 使用 alpha/beta 调整正样本区域的阈值
        pos_threshold = 1.0 - (0.5 ** (1.0 / self.beta)) if self.beta > 0 else 0.5
        pos_mask = (gt_heatmap > pos_threshold).float()
        weights = 1.0 + pos_mask * (self.pos_weight - 1.0)

        weighted_bce = bce * weights

        return weighted_bce.mean()


class WeightedMSELoss(nn.Module):
    """
    加权MSE损失

    对正样本区域（heatmap > 0.5）给予更高权重。
    """

    def __init__(self, pos_weight: float = 10.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        pred_logits: torch.Tensor,
        gt_heatmap: torch.Tensor
    ) -> torch.Tensor:
        if pred_logits.dim() == 3:
            pred_logits = pred_logits.unsqueeze(1)
        if gt_heatmap.dim() == 4:
            gt_heatmap = gt_heatmap.squeeze(1)

        # 使用sigmoid，但detach以避免autocast问题
        with torch.cuda.amp.autocast(enabled=False):
            pred = torch.sigmoid(pred_logits.float()).squeeze(1)

        # 正样本区域权重更高
        pos_mask = (gt_heatmap > 0.5).float()
        weights = 1.0 + pos_mask * (self.pos_weight - 1.0)

        loss = ((pred - gt_heatmap) ** 2) * weights
        return loss.mean()


class CombinedHeatmapLoss(nn.Module):
    """
    组合热力图损失

    结合Focal Loss和MSE，兼顾定位精度和整体热力图质量。

    参数:
        focal_weight: Focal损失权重
        mse_weight: MSE损失权重
    """

    def __init__(
        self,
        focal_weight: float = 1.0,
        mse_weight: float = 1.0
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.mse_weight = mse_weight
        self.focal_loss = HeatmapFocalLoss()
        self.mse_loss = WeightedMSELoss()

    def forward(
        self,
        pred_logits: torch.Tensor,
        gt_heatmap: torch.Tensor
    ) -> torch.Tensor:
        focal = self.focal_loss(pred_logits, gt_heatmap)
        mse = self.mse_loss(pred_logits, gt_heatmap)
        return self.focal_weight * focal + self.mse_weight * mse


class MSEHeatmapLoss(nn.Module):
    """标准MSE热力图损失"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_logits: torch.Tensor,
        gt_heatmap: torch.Tensor
    ) -> torch.Tensor:
        if pred_logits.dim() == 3:
            pred_logits = pred_logits.unsqueeze(1)
        if gt_heatmap.dim() == 3:
            gt_heatmap = gt_heatmap.unsqueeze(1)

        pred = torch.sigmoid(pred_logits)
        return F.mse_loss(pred, gt_heatmap)


# 损失函数注册表
LOSS_REGISTRY = {
    'heatmap_focal': HeatmapFocalLoss,
    'mse': MSEHeatmapLoss,
    'weighted_mse': WeightedMSELoss,
    'combined': CombinedHeatmapLoss
}


def build_loss(loss_type: str = 'heatmap_focal', **kwargs):
    """
    根据名称构建损失函数

    参数:
        loss_type: 损失函数类型
        **kwargs: 传递给损失函数的参数

    返回:
        损失函数实例
    """
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"未知的损失函数类型: {loss_type}，可用: {list(LOSS_REGISTRY.keys())}")

    return LOSS_REGISTRY[loss_type](**kwargs)


if __name__ == '__main__':
    # 测试损失函数
    import torch

    B, H, W = 2, 64, 64
    pred_logits = torch.randn(B, 1, H, W)

    # 创建高斯热力图
    gt_heatmap = torch.zeros(B, H, W)
    for b in range(B):
        cx, cy = 32, 32
        for i in range(H):
            for j in range(W):
                gt_heatmap[b, i, j] = torch.exp(
                    torch.tensor(-((j - cx)**2 + (i - cy)**2) / (2 * 5**2))
                )

    # 测试
    focal_loss = HeatmapFocalLoss()
    loss1 = focal_loss(pred_logits, gt_heatmap)
    print(f"HeatmapFocalLoss: {loss1.item():.4f}")

    weighted_mse = WeightedMSELoss()
    loss2 = weighted_mse(pred_logits, gt_heatmap)
    print(f"WeightedMSELoss: {loss2.item():.4f}")

    print("\n损失函数测试通过！")
