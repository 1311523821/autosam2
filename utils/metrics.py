"""
评估指标 (Evaluation Metrics)

与LVNet保持一致的NUDT数据集评估指标。

指标说明：
- IoU: 全局累积的Intersection over Union
- nIoU: 每张图片IoU的平均值（Normalized IoU）
- Pd: 检测概率，基于连通域质心距离<3像素的目标检测率
- Fa: 虚警率，虚警像素数/总像素数
"""

import numpy as np
from skimage import measure
from typing import Tuple, Dict, Optional


class NUDT_Metrics:
    """
    NUDT数据集评估指标

    与/root/e2e/LVNet_ablation.py中的NUDT_Metrics完全一致。

    使用方法：
        metrics = NUDT_Metrics(thre=0.5)
        for pred, label in zip(predictions, labels):
            metrics.update(pred, label)
        iou, niou, pd, fa = metrics.get()
    """

    def __init__(self, thre: float = 0.5):
        """
        初始化指标

        Args:
            thre: 二值化阈值
        """
        self.thre = thre
        self.total_inter = 0   # 累积交集
        self.total_union = 0   # 累积并集
        self.niou_sum = 0      # nIoU累加
        self.niou_count = 0    # nIoU计数
        self.FA0 = 0           # 虚警像素数
        self.PD0 = 0           # 检测到的目标数
        self.target_count = 0  # 总目标数
        self.pixel_count = 0   # 总像素数

    def update(self, preds: np.ndarray, labels: np.ndarray):
        """
        更新指标

        Args:
            preds: 预测结果，形状任意，值域[0,1]或二值
            labels: 标签，形状与preds相同
        """
        # 二值化
        predits = (preds > self.thre).astype(np.int64)
        labelss = labels.astype(np.int64)

        # 计算交集和并集
        inter = np.sum(predits & labelss)
        union = np.sum(predits | labelss)
        self.total_inter += inter
        self.total_union += union

        # 计算每张图片的IoU（用于nIoU）
        if union == 0:
            self.niou_sum += 1.0  # 空图片认为IoU=1
        else:
            self.niou_sum += (inter / union)
        self.niou_count += 1

        # 连通域分析（用于Pd和Fa计算）
        image_cc = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image_cc)
        label_cc = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label_cc)

        self.target_count += len(coord_label)
        coord_image_list = list(coord_image)

        # 计算Pd：检测到的目标数
        # 如果预测连通域的质心与GT目标质心距离<3像素，则认为检测成功
        for i in range(len(coord_label)):
            centroid_label = np.array(coord_label[i].centroid)
            for m in range(len(coord_image_list)):
                centroid_image = np.array(coord_image_list[m].centroid)
                distance = np.linalg.norm(centroid_image - centroid_label)
                if distance < 3:
                    self.PD0 += 1
                    del coord_image_list[m]
                    break

        # 计算Fa：虚警像素数
        dismatch_areas = [prop.area for prop in coord_image_list]
        self.FA0 += np.sum(dismatch_areas) if len(dismatch_areas) > 0 else 0
        self.pixel_count += predits.size

    def merge(self, other: 'NUDT_Metrics'):
        """
        合并另一个metrics对象

        Args:
            other: 另一个NUDT_Metrics对象
        """
        self.total_inter += other.total_inter
        self.total_union += other.total_union
        self.niou_sum += other.niou_sum
        self.niou_count += other.niou_count
        self.FA0 += other.FA0
        self.PD0 += other.PD0
        self.target_count += other.target_count
        self.pixel_count += other.pixel_count

    def get(self) -> Tuple[float, float, float, float]:
        """
        获取最终指标

        Returns:
            iou: 全局IoU
            niou: 平均每图IoU
            pd: 检测概率
            fa: 虚警率
        """
        iou = self.total_inter / (self.total_union + 1e-6)
        niou = self.niou_sum / (self.niou_count + 1e-6)
        pd = self.PD0 / (self.target_count + 1e-6)
        fa = self.FA0 / (self.pixel_count + 1e-6)
        return iou, niou, pd, fa

    def compute(self) -> Dict[str, float]:
        """
        计算并返回字典格式的指标（兼容验证脚本）

        Returns:
            包含iou, niou, pd, fa的字典
        """
        iou, niou, pd, fa = self.get()
        return {
            'iou': iou,
            'niou': niou,
            'pd': pd,
            'fa': fa
        }

    def reset(self):
        """重置所有指标"""
        self.total_inter = 0
        self.total_union = 0
        self.niou_sum = 0
        self.niou_count = 0
        self.FA0 = 0
        self.PD0 = 0
        self.target_count = 0
        self.pixel_count = 0


def compute_iou(pred, target, threshold: float = 0.5) -> float:
    """
    计算单张图片的IoU

    Args:
        pred: 预测结果，形状(H, W)，支持numpy数组或torch张量
        target: 标签，形状(H, W)，支持numpy数组或torch张量
        threshold: 二值化阈值

    Returns:
        IoU值
    """
    # 转换为numpy数组（如果输入是torch张量）
    if hasattr(pred, 'cpu'):  # torch tensor
        pred = pred.cpu().numpy()
    if hasattr(target, 'cpu'):  # torch tensor
        target = target.cpu().numpy()

    pred_binary = (pred > threshold).astype(np.float32)
    target = target.astype(np.float32)

    intersection = np.sum(pred_binary * target)
    union = np.sum(pred_binary) + np.sum(target) - intersection

    return float(intersection / (union + 1e-6))


def compute_dice(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """
    计算Dice系数

    Args:
        pred: 预测结果
        target: 标签
        threshold: 二值化阈值

    Returns:
        Dice值
    """
    pred_binary = (pred > threshold).astype(np.float32)
    target = target.astype(np.float32)

    intersection = np.sum(pred_binary * target)

    return float(2 * intersection / (np.sum(pred_binary) + np.sum(target) + 1e-6))
