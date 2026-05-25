"""
工具函数模块
包含评估指标、热力图生成、损失函数等
"""

from .metrics import NUDT_Metrics, compute_iou, compute_dice
from .heatmap import (
    generate_gaussian_heatmap,
    generate_multi_target_heatmap,
    extract_point_from_heatmap,
    extract_multiple_points_from_heatmap,
    get_mask_center,
    get_mask_bbox
)
from .losses import (
    HeatmapFocalLoss,
    MSEHeatmapLoss,
    CombinedHeatmapLoss,
    build_loss,
    LOSS_REGISTRY
)

__all__ = [
    # 评估指标
    'NUDT_Metrics',
    'compute_iou',
    'compute_dice',
    # 热力图工具
    'generate_gaussian_heatmap',
    'generate_multi_target_heatmap',
    'extract_point_from_heatmap',
    'extract_multiple_points_from_heatmap',
    'get_mask_center',
    'get_mask_bbox',
    # 损失函数
    'HeatmapFocalLoss',
    'MSEHeatmapLoss',
    'CombinedHeatmapLoss',
    'build_loss',
    'LOSS_REGISTRY'
]
