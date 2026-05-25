#!/usr/bin/env python
"""
引导头评估脚本 (Guidance Head Evaluation Script)

评估训练好的引导头模型，输出定位准确率和可视化结果。

用法:
    python evaluate_guidance.py --checkpoint checkpoints/guidance_head/best.pth
    python evaluate_guidance.py --checkpoint best.pth --visualize
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, '/root/autosam2')

from models.guidance_head import GuidanceHead
from data.dataset import VideoHeatmapDataset, heatmap_collate_fn, get_heatmap_dataloader
from utils.heatmap import extract_point_from_heatmap, get_mask_center


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估引导头模型')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--data-root', type=str, default='/root/DataBscan',
                        help='数据根目录')
    parser.add_argument('--test-list', type=str, default='/root/e2e/test1.txt',
                        help='测试集列表文件')
    parser.add_argument('--target-label', type=str, default='uuv',
                        help='目标标签')
    parser.add_argument('--clip-len', type=int, default=4,
                        help='时序帧数')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='批大小')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载工作进程数')
    parser.add_argument('--heatmap-sigma', type=float, default=3.0,
                        help='高斯热力图标准差')
    parser.add_argument('--threshold-radius', type=int, default=10,
                        help='命中判定半径（像素）')
    parser.add_argument('--output', type=str, default='results/guidance_eval',
                        help='输出目录')
    parser.add_argument('--visualize', action='store_true',
                        help='是否保存可视化结果')
    parser.add_argument('--device', type=str, default='cuda',
                        help='评估设备')

    return parser.parse_args()


def evaluate_model(
    model,
    dataloader,
    device,
    threshold_radius=10,
    output_dir=None,
    visualize=False
):
    """
    评估模型

    返回:
        metrics: 评估指标字典
        per_sample: 每个样本的详细结果
    """
    model.eval()

    all_distances = []
    all_confidences = []
    all_hits = []
    per_sample_results = []

    vis_dir = None
    if visualize and output_dir:
        vis_dir = Path(output_dir) / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (frames, heatmap_gt, masks, centers) in enumerate(tqdm(dataloader, desc="评估中")):
            frames = frames.to(device)
            heatmap_gt = heatmap_gt.to(device)
            masks = masks.to(device)
            centers = centers.to(device)

            # 前向传播
            pred_logits = model(frames)
            pred_heatmap = torch.sigmoid(pred_logits)

            B = frames.shape[0]

            for i in range(B):
                # 提取预测点
                pred_map = pred_heatmap[i, 0]  # (H, W)
                flat_idx = torch.argmax(pred_map.view(-1))
                H, W = pred_map.shape
                pred_y = flat_idx // W
                pred_x = flat_idx % W
                confidence = pred_map.view(-1)[flat_idx].item()

                # GT中心
                gt_x, gt_y = centers[i, 0].item(), centers[i, 1].item()

                # 计算距离
                distance = np.sqrt((pred_x.item() - gt_x)**2 + (pred_y.item() - gt_y)**2)
                hit = distance <= threshold_radius

                all_distances.append(distance)
                all_confidences.append(confidence)
                all_hits.append(hit)

                # 记录详细结果
                per_sample_results.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'pred_x': pred_x.item(),
                    'pred_y': pred_y.item(),
                    'gt_x': gt_x,
                    'gt_y': gt_y,
                    'distance': distance,
                    'hit': hit,
                    'confidence': confidence
                })

                # 可视化
                if visualize and vis_dir and batch_idx < 5:  # 只保存前5个batch
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    # 输入帧（取最后一帧）
                    last_frame = frames[i, 0, -1].cpu().numpy()
                    axes[0].imshow(last_frame, cmap='gray')
                    axes[0].set_title('Input Frame (Last)')
                    axes[0].axis('off')

                    # 预测热力图
                    pred_np = pred_map.cpu().numpy()
                    axes[1].imshow(pred_np, cmap='jet')
                    axes[1].scatter([pred_x.item()], [pred_y.item()], c='red', s=100, marker='x', label='Pred')
                    axes[1].scatter([gt_x], [gt_y], c='green', s=100, marker='+', label='GT')
                    axes[1].set_title(f'Predicted Heatmap (dist={distance:.1f}px)')
                    axes[1].legend()
                    axes[1].axis('off')

                    # GT热力图
                    gt_np = heatmap_gt[i].cpu().numpy()
                    axes[2].imshow(gt_np, cmap='jet')
                    axes[2].set_title('GT Heatmap')
                    axes[2].axis('off')

                    plt.tight_layout()
                    save_path = vis_dir / f'batch{batch_idx}_sample{i}.png'
                    plt.savefig(save_path, dpi=100)
                    plt.close()

    # 计算汇总指标
    metrics = {
        'num_samples': len(all_distances),
        'hit_rate': sum(all_hits) / len(all_hits) if all_hits else 0,
        'mean_distance': np.mean(all_distances) if all_distances else 0,
        'std_distance': np.std(all_distances) if all_distances else 0,
        'median_distance': np.median(all_distances) if all_distances else 0,
        'mean_confidence': np.mean(all_confidences) if all_confidences else 0,
        'threshold_radius': threshold_radius,
        # 不同阈值的命中率
        'hit_rate_5': sum(d <= 5 for d in all_distances) / len(all_distances) if all_distances else 0,
        'hit_rate_10': sum(d <= 10 for d in all_distances) / len(all_distances) if all_distances else 0,
        'hit_rate_15': sum(d <= 15 for d in all_distances) / len(all_distances) if all_distances else 0,
        'hit_rate_20': sum(d <= 20 for d in all_distances) / len(all_distances) if all_distances else 0,
    }

    return metrics, per_sample_results


def main():
    """主函数"""
    args = parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"\n加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # 从检查点推断模型参数
    model = GuidanceHead(
        in_chans=1,
        embed_dim=24,  # 默认值
        hidden_dim=12,
        num_frame=args.clip_len
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型加载成功，来自 epoch {checkpoint.get('epoch', 'unknown')}")

    # 创建测试数据集
    print(f"\n加载测试数据集: {args.test_list}")
    test_loader = get_heatmap_dataloader(
        txt_path=args.test_list,
        root_dir=args.data_root,
        clip_len=args.clip_len,
        target_label=args.target_label,
        is_train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        heatmap_sigma=args.heatmap_sigma
    )

    # 评估
    print("\n开始评估...")
    metrics, per_sample = evaluate_model(
        model, test_loader, device,
        threshold_radius=args.threshold_radius,
        output_dir=output_dir,
        visualize=args.visualize
    )

    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"样本数量: {metrics['num_samples']}")
    print(f"\n定位准确率:")
    print(f"  Hit Rate@5px:  {metrics['hit_rate_5']:.2%}")
    print(f"  Hit Rate@10px: {metrics['hit_rate_10']:.2%}")
    print(f"  Hit Rate@15px: {metrics['hit_rate_15']:.2%}")
    print(f"  Hit Rate@20px: {metrics['hit_rate_20']:.2%}")
    print(f"\n距离统计:")
    print(f"  平均距离: {metrics['mean_distance']:.2f} px")
    print(f"  标准差: {metrics['std_distance']:.2f} px")
    print(f"  中位数: {metrics['median_distance']:.2f} px")
    print(f"\n置信度:")
    print(f"  平均置信度: {metrics['mean_confidence']:.4f}")
    print("=" * 60)

    # 保存结果
    results = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint': args.checkpoint,
        'config': vars(args),
        'metrics': metrics,
        'per_sample_results': per_sample[:100]  # 只保存前100个样本详情
    }

    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {results_path}")

    if args.visualize:
        print(f"可视化结果保存在: {output_dir / 'visualizations'}")

    # 判断是否达到目标
    if metrics['hit_rate_10'] >= 0.90:
        print("\n🎉 模型达到目标！Hit Rate@10 >= 90%")
    else:
        print(f"\n⚠️ 模型未达目标，Hit Rate@10 = {metrics['hit_rate_10']:.2%} < 90%")


if __name__ == '__main__':
    main()
