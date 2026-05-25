#!/usr/bin/env python
"""
阶段二：引导头可视化脚本

可视化引导头的预测结果，包括：
- 每帧的灰度图 + GT中心点 + 预测点
- 引导热力图
- 预测误差统计

用法:
    python scripts/visualize_guidance.py \
        --guidance-ckpt checkpoints/experiments/exp6_embed64/.../best.pth \
        --test-list /root/e2e/test1.txt \
        --output results/guidance_visualization
"""

import os
import sys
import argparse
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # 非交互式后端

sys.path.insert(0, '/root/autosam2')
sys.path.insert(0, '/root/e2e')

from models.guidance_head import build_guidance_head


def parse_args():
    parser = argparse.ArgumentParser(description='引导头可视化')
    parser.add_argument('--guidance-ckpt', type=str,
                        default='checkpoints/experiments/exp6_embed64/enc(stsf)_embed(64)_hidden(32)_opt(adamw)_lr(5e-4)_loss(focal_a2)_sigma(10)_kfold(5)/fold_1/best.pth',
                        help='引导头权重路径')
    parser.add_argument('--data-root', type=str, default='/root/DataBscan',
                        help='数据根目录')
    parser.add_argument('--test-list', type=str, default='/root/e2e/test1.txt',
                        help='测试列表文件')
    parser.add_argument('--output', type=str, default='results/guidance_visualization',
                        help='输出目录')
    parser.add_argument('--target-label', type=str, default='uuv',
                        help='目标标签')
    parser.add_argument('--max-videos', type=int, default=None,
                        help='最大处理视频数')
    parser.add_argument('--target-size', type=int, default=256,
                        help='引导头输入分辨率')
    parser.add_argument('--thresholds', type=str, default='5,10',
                        help='Hit Rate阈值列表，逗号分隔')
    return parser.parse_args()


def load_grayscale_frames(video_folder: str, target_size: int = 256):
    """加载灰度帧并resize"""
    img_files = sorted([
        f for f in os.listdir(video_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])
    frames = []
    orig_shapes = []
    for img_file in img_files:
        img_path = os.path.join(video_folder, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            orig_shapes.append(img.shape)
            img = cv2.resize(img, (target_size, target_size))
            frames.append(img.astype(np.float32) / 255.0)
    return np.stack(frames) if frames else None, orig_shapes, img_files


def load_gt_centers(video_folder: str, frame_files: list, target_label: str, target_size: int, orig_shapes: list):
    """加载GT中心点并转换到target_size坐标系"""
    centers = []

    for i, img_file in enumerate(frame_files):
        json_file = img_file.rsplit('.', 1)[0] + '.json'
        json_path = os.path.join(video_folder, json_file)

        if not os.path.exists(json_path):
            centers.append(None)
            continue

        with open(json_path, 'r') as f:
            annotation = json.load(f)

        center = None
        for shape in annotation.get('shapes', []):
            if shape.get('label') == target_label:
                points = np.array(shape['points'])
                center_x = np.mean(points[:, 0])
                center_y = np.mean(points[:, 1])

                # 转换到target_size坐标系
                if i < len(orig_shapes):
                    orig_h, orig_w = orig_shapes[i][:2]
                    scale_x = target_size / orig_w
                    scale_y = target_size / orig_h
                    center = (center_x * scale_x, center_y * scale_y)
                break

        centers.append(center)

    return centers


def visualize_frame(
    frame: np.ndarray,
    heatmap: np.ndarray,
    gt_center: tuple,
    pred_point: tuple,
    error: float,
    frame_idx: int,
    save_path: str
):
    """可视化单帧结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 左图：灰度帧 + 点
    axes[0].imshow(frame, cmap='gray')
    axes[0].set_title(f'Frame {frame_idx}')
    if gt_center is not None:
        axes[0].plot(gt_center[0], gt_center[1], 'go', markersize=10, label='GT Center')
    axes[0].plot(pred_point[0], pred_point[1], 'ro', markersize=10, label='Predicted')
    axes[0].legend()
    axes[0].axis('off')

    # 中图：热力图
    axes[1].imshow(heatmap, cmap='hot')
    axes[1].set_title('Guidance Heatmap')
    axes[1].plot(pred_point[0], pred_point[1], 'bo', markersize=8)
    axes[1].axis('off')

    # 右图：叠加
    axes[2].imshow(frame, cmap='gray', alpha=0.7)
    axes[2].imshow(heatmap, cmap='hot', alpha=0.3)
    axes[2].set_title(f'Overlay (Error: {error:.2f}px)')
    if gt_center is not None:
        axes[2].plot(gt_center[0], gt_center[1], 'go', markersize=10, label='GT')
    axes[2].plot(pred_point[0], pred_point[1], 'ro', markersize=10, label='Pred')
    axes[2].legend()
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()

    print("=" * 60)
    print("引导头可视化")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载引导头模型
    print(f"\n加载引导头: {args.guidance_ckpt}")
    guidance_config = {
        'in_chans': 1,
        'embed_dim': 64,
        'hidden_dim': 32,
        'num_frame': 4
    }

    guidance_head = build_guidance_head(guidance_config)
    checkpoint = torch.load(args.guidance_ckpt, map_location='cuda')
    if 'model_state_dict' in checkpoint:
        guidance_head.load_state_dict(checkpoint['model_state_dict'])
    else:
        guidance_head.load_state_dict(checkpoint)
    guidance_head = guidance_head.cuda()
    guidance_head.eval()
    print("引导头加载成功！")

    # 加载测试列表
    with open(args.test_list, 'r') as f:
        video_folders = [line.strip().replace('\\', '/') for line in f if line.strip()]

    if args.max_videos:
        video_folders = video_folders[:args.max_videos]

    print(f"测试视频数: {len(video_folders)}")

    # 解析阈值列表
    thresholds = [int(t) for t in args.thresholds.split(',')]
    print(f"Hit Rate阈值: {thresholds}")

    # 统计变量
    all_errors = []
    all_hits = {t: [] for t in thresholds}
    video_stats = []

    # 处理每个视频
    for video_idx, video_folder in enumerate(tqdm(video_folders, desc="可视化中")):
        video_path = os.path.join(args.data_root, video_folder)

        if not os.path.exists(video_path):
            print(f"警告: 视频不存在 {video_path}")
            continue

        # 创建视频输出目录
        video_output_dir = output_dir / f'video_{video_idx}'
        video_output_dir.mkdir(exist_ok=True)

        # 加载帧
        frames, orig_shapes, frame_files = load_grayscale_frames(video_path, args.target_size)
        if frames is None or len(frames) == 0:
            continue

        # 加载GT中心点
        gt_centers = load_gt_centers(video_path, frame_files, args.target_label,
                                      args.target_size, orig_shapes)

        # 处理每个clip
        video_errors = []
        video_hits = {t: [] for t in thresholds}

        T = len(frames)
        for start_idx in range(0, T, 4):  # 每4帧一个clip
            # 提取4帧clip
            end_idx = min(start_idx + 4, T)
            clip = frames[start_idx:end_idx]

            if clip.shape[0] < 4:
                pad = np.repeat(clip[-1:], 4 - clip.shape[0], axis=0)
                clip = np.concatenate([clip, pad], axis=0)

            # 转换为tensor (B, 1, D, H, W)
            clip_tensor = torch.from_numpy(clip).float().unsqueeze(0).unsqueeze(0).cuda()

            # 推理
            with torch.no_grad():
                heatmap = guidance_head(clip_tensor)
                prompt_point = guidance_head.get_prompt_point(heatmap)

            # 获取预测点
            pred_point = prompt_point[0].cpu().numpy()  # (x, y)
            heatmap_np = torch.sigmoid(heatmap[0, 0]).cpu().numpy()

            # 可视化中间帧（start_idx + 2）
            vis_frame_idx = start_idx + 2
            if vis_frame_idx >= T:
                vis_frame_idx = T - 1

            gt_center = gt_centers[vis_frame_idx] if vis_frame_idx < len(gt_centers) else None

            # 计算误差
            error = None
            if gt_center is not None:
                error = np.sqrt((pred_point[0] - gt_center[0])**2 +
                               (pred_point[1] - gt_center[1])**2)
                video_errors.append(error)
                all_errors.append(error)
                for t in thresholds:
                    is_hit = error <= t
                    video_hits[t].append(is_hit)
                    all_hits[t].append(is_hit)

            # 保存可视化
            save_path = str(video_output_dir / f'clip_{start_idx:04d}.png')
            visualize_frame(
                frames[vis_frame_idx],
                heatmap_np,
                gt_center,
                pred_point,
                error if error is not None else 0.0,
                vis_frame_idx,
                save_path
            )

        # 视频统计
        if video_errors:
            video_stat = {
                'video_idx': video_idx,
                'video_folder': video_folder,
                'num_clips': len(video_errors),
                'mean_error': float(np.mean(video_errors)),
                'std_error': float(np.std(video_errors)),
            }
            for t in thresholds:
                video_stat[f'hit_rate@{t}'] = float(np.mean(video_hits[t]))
            video_stats.append(video_stat)

    # 生成汇总图
    if all_errors:
        # 误差分布直方图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 误差分布
        axes[0].hist(all_errors, bins=50, edgecolor='black', alpha=0.7)
        colors = ['r', 'g', 'b', 'orange']
        for i, t in enumerate(thresholds):
            color = colors[i % len(colors)]
            axes[0].axvline(x=t, color=color, linestyle='--',
                           label=f'Threshold={t}px')
        axes[0].set_xlabel('Error (pixels)')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Error Distribution (Mean: {np.mean(all_errors):.2f}px)')
        axes[0].legend()

        # 多个 Hit Rate 柱状图
        x_labels = [f'@{t}' for t in thresholds]
        hit_rates = [np.mean(all_hits[t]) for t in thresholds]
        bars = axes[1].bar(x_labels, hit_rates, color=['green', 'blue', 'orange'][:len(thresholds)])
        axes[1].set_ylabel('Rate')
        axes[1].set_title('Hit Rate at Different Thresholds')
        axes[1].set_ylim(0, 1)
        # 在柱子上显示数值
        for bar, rate in zip(bars, hit_rates):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{rate*100:.1f}%', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(str(output_dir / 'summary.png'), dpi=150)
        plt.close(fig)

        print(f"\n汇总图保存到: {output_dir / 'summary.png'}")

    # 保存统计报告
    report = {
        'config': {
            'guidance_ckpt': args.guidance_ckpt,
            'data_root': args.data_root,
            'test_list': args.test_list,
            'target_size': args.target_size,
            'thresholds': thresholds
        },
        'overall_stats': {
            'total_clips': len(all_errors),
            'mean_error': float(np.mean(all_errors)) if all_errors else 0,
            'std_error': float(np.std(all_errors)) if all_errors else 0,
        },
        'per_video_stats': video_stats
    }
    # 添加各阈值的Hit Rate
    for t in thresholds:
        report['overall_stats'][f'hit_rate@{t}'] = float(np.mean(all_hits[t])) if all_hits[t] else 0

    with open(output_dir / 'visualization_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    # 打印结果
    print("\n" + "=" * 60)
    print("可视化完成")
    print("=" * 60)
    print(f"总clip数: {len(all_errors)}")
    print(f"平均误差: {np.mean(all_errors):.2f} ± {np.std(all_errors):.2f} 像素")
    for t in thresholds:
        print(f"Hit Rate@{t}: {np.mean(all_hits[t])*100:.1f}%")
    print(f"\n输出目录: {output_dir}")
    print(f"  - summary.png: 汇总统计图")
    print(f"  - visualization_report.json: 详细报告")
    print(f"  - video_X/: 每个视频的可视化")


if __name__ == '__main__':
    main()
