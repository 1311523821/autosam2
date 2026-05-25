#!/usr/bin/env python
"""
阶段一：SAM 2 可行性验证脚本

验证SAM 2在声呐数据上跟踪小目标的能力。
支持多个模型大小的性能对比（精度和速度）。

用法:
    python scripts/validate_sam2.py --data-root /root/DataBscan --test-list /root/e2e/test1.txt
    python scripts/validate_sam2.py --benchmark  # 启用FPS计时
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import torch
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, '/root/autosam2')
sys.path.insert(0, '/root/e2e')

from utils.metrics import NUDT_Metrics, compute_iou
from utils.heatmap import get_mask_center


def parse_args():
    parser = argparse.ArgumentParser(description='验证SAM 2在声呐数据上的表现')
    parser.add_argument('--data-root', type=str, default='/root/DataBscan',
                        help='数据根目录')
    parser.add_argument('--test-list', type=str, default='/root/e2e/test1.txt',
                        help='测试列表文件路径')
    parser.add_argument('--sam2-ckpt', type=str,
                        default='/root/autosam2/checkpoints/sam2.1_hiera_large.pt',
                        help='SAM 2权重路径')
    parser.add_argument('--sam2-config', type=str,
                        default='sam2.1_hiera_l.yaml',
                        help='SAM 2配置文件')
    parser.add_argument('--output', type=str, default='results/phase1_validation',
                        help='输出目录')
    parser.add_argument('--target-label', type=str, default='uuv',
                        help='目标标签')
    parser.add_argument('--max-videos', type=int, default=None,
                        help='最大处理视频数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='运行设备')
    parser.add_argument('--visualize', action='store_true',
                        help='保存可视化结果')
    parser.add_argument('--benchmark', action='store_true',
                        help='启用FPS计时（性能测试模式）')
    return parser.parse_args()


def load_video_data(folder_path: str, target_label: str):
    """Load video frames and ground truth masks."""
    valid_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    frames, masks, centers = [], [], []

    for name in valid_files:
        # Load frame
        img = cv2.imread(os.path.join(folder_path, name), cv2.IMREAD_GRAYSCALE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        frames.append(img_rgb)

        # Load GT mask
        json_name = os.path.splitext(name)[0] + '.json'
        json_path = os.path.join(folder_path, json_name)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for s in data.get('shapes', []):
                    if s['label'] == target_label:
                        points = np.array(s['points'], dtype=np.float32)
                        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
            except Exception as e:
                print(f"Error loading {json_path}: {e}")

        masks.append(mask)

        # Get center point
        center = get_mask_center(torch.from_numpy(mask))
        centers.append(center if center is not None else None)

    return frames, masks, centers, valid_files


def init_sam2_predictor(checkpoint_path: str, config_path: str, device: str):
    """Initialize SAM 2 video predictor."""
    try:
        from sam2.build_sam import build_sam2_video_predictor

        # SAM 2 expects config path relative to sam2 package configs directory
        # e.g., 'configs/sam2.1/sam2.1_hiera_l.yaml'
        if not config_path.startswith('configs/'):
            config_path = f'configs/sam2.1/{config_path}'

        predictor = build_sam2_video_predictor(config_path, checkpoint_path, device=device)
        print(f"SAM 2 predictor loaded from {checkpoint_path}")
        return predictor
    except ImportError as e:
        print(f"Error importing SAM 2: {e}")
        print("Make sure sam2 is installed: pip install sam2")
        return None
    except Exception as e:
        print(f"Error loading SAM 2: {e}")
        return None


def validate_video(
    predictor,
    folder_path: str,
    frame_names: list,
    gt_masks: list,
    gt_centers: list,
    video_idx: int,
    output_dir: str = None,
    visualize: bool = False,
    benchmark: bool = False
):
    """
    验证SAM 2在单个视频上的表现

    参数:
        predictor: SAM 2视频预测器
        folder_path: 视频文件夹路径
        frame_names: 帧文件名列表
        gt_masks: GT掩码列表
        gt_centers: 中心点列表
        video_idx: 视频索引
        output_dir: 输出目录
        visualize: 是否保存可视化
        benchmark: 是否启用FPS计时

    返回:
        包含每帧指标和速度信息的字典
    """
    import numpy as np

    # 计时变量
    total_time = 0.0
    frame_times = []

    # SAM 2需要文件夹路径，不是已加载的帧
    init_start = time.time()
    inference_state = predictor.init_state(folder_path)
    init_time = time.time() - init_start

    # 找到第一个有有效中心点的帧
    prompt_frame_idx = 0
    prompt_point = None

    for i, center in enumerate(gt_centers):
        if center is not None:
            prompt_frame_idx = i
            prompt_point = np.array([[center[0], center[1]]])
            break

    if prompt_point is None:
        print(f"视频 {video_idx}: 未找到有效目标")
        return None

    # 添加Prompt点
    prompt_start = time.time()
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=prompt_frame_idx,
        obj_id=1,
        points=prompt_point,
        labels=np.array([1], dtype=np.int32)
    )
    prompt_time = time.time() - prompt_start

    # 视频传播跟踪
    results = []
    video_segments = {}

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        frame_start = time.time()

        mask_logits = out_mask_logits[0]  # (1, H, W)
        pred_mask = (mask_logits > 0).cpu().numpy().squeeze()

        gt_mask = gt_masks[out_frame_idx]

        # 计算指标
        iou = compute_iou(
            torch.from_numpy(pred_mask).float(),
            torch.from_numpy(gt_mask).float()
        )

        frame_time = time.time() - frame_start
        if benchmark:
            frame_times.append(frame_time)
            total_time += frame_time

        results.append({
            'frame_idx': out_frame_idx,
            'iou': iou,
            'has_target': gt_mask.sum() > 0,
            'frame_time_ms': frame_time * 1000 if benchmark else None
        })

        video_segments[out_frame_idx] = pred_mask

    # Visualize if requested
    if visualize and output_dir:
        vis_dir = os.path.join(output_dir, 'visualizations', f'video_{video_idx}')
        os.makedirs(vis_dir, exist_ok=True)

        # Load frames for visualization
        for frame_idx, frame_name in enumerate(frame_names):
            if frame_idx not in video_segments:
                continue

            frame_path = os.path.join(folder_path, frame_name)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            pred_mask = video_segments[frame_idx]
            gt_mask = gt_masks[frame_idx] if frame_idx < len(gt_masks) else None

            if gt_mask is not None:
                # Draw GT mask (green)
                gt_overlay = np.zeros_like(frame)
                gt_overlay[gt_mask > 0] = [0, 255, 0]
                frame = cv2.addWeighted(frame, 0.7, gt_overlay, 0.3, 0)

            # Draw predicted mask (red)
            pred_overlay = np.zeros_like(frame)
            pred_overlay[pred_mask > 0] = [0, 0, 255]
            frame = cv2.addWeighted(frame, 0.7, pred_overlay, 0.3, 0)

            # Draw center points
            if frame_idx < len(gt_centers) and gt_centers[frame_idx] is not None:
                cx, cy = gt_centers[frame_idx]
                cv2.circle(frame, (int(cx), int(cy)), 5, (255, 255, 0), -1)

            # Add frame info
            iou = results[frame_idx]['iou'] if frame_idx < len(results) else 0
            cv2.putText(frame, f'Frame {frame_idx}, IoU: {iou:.3f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imwrite(os.path.join(vis_dir, f'frame_{frame_idx:04d}.png'), frame)

    # 返回结果包含速度信息和预测结果
    speed_info = {
        'init_time_s': init_time,
        'prompt_time_s': prompt_time,
        'total_time_s': total_time,
        'num_frames': len(frame_times),
        'avg_frame_time_ms': np.mean(frame_times) * 1000 if frame_times else 0,
        'fps': len(frame_times) / total_time if total_time > 0 else 0
    }

    return {
        'results': results,
        'video_segments': video_segments,  # 返回预测mask用于计算NUDT指标
        'speed': speed_info if benchmark else None
    }


def main():
    """主函数"""
    args = parse_args()

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载测试视频列表
    with open(args.test_list, 'r', encoding='utf-8') as f:
        video_folders = [
            os.path.join(args.data_root, line.strip().replace('\\', '/'))
            for line in f.readlines()
            if line.strip()
        ]

    if args.max_videos:
        video_folders = video_folders[:args.max_videos]

    print(f"发现 {len(video_folders)} 个视频待验证")
    print(f"Benchmark模式: {'开启' if args.benchmark else '关闭'}")

    # 初始化SAM 2预测器
    predictor = init_sam2_predictor(args.sam2_ckpt, args.sam2_config, args.device)
    if predictor is None:
        print("SAM 2预测器初始化失败")
        return

    # 初始化指标
    all_metrics = NUDT_Metrics()
    all_results = []
    all_speed_info = []  # 存储所有视频的速度信息

    # 处理每个视频
    for video_idx, folder_path in enumerate(tqdm(video_folders, desc="验证中")):
        if not os.path.isdir(folder_path):
            continue

        # 加载GT掩码和中心点（帧由SAM 2直接加载）
        _, masks, centers, frame_names = load_video_data(folder_path, args.target_label)

        if len(frame_names) == 0:
            continue

        # 验证
        result = validate_video(
            predictor, folder_path, frame_names, masks, centers,
            video_idx, str(output_dir), args.visualize, args.benchmark
        )

        if result is None:
            continue

        frame_results = result['results']
        video_segments = result['video_segments']  # 获取预测的mask
        speed_info = result['speed']

        # 聚合指标 - 使用实际的预测结果
        for frame_idx, pred_mask in video_segments.items():
            if frame_idx < len(masks):
                # 使用numpy数组直接调用metrics.update()
                all_metrics.update(
                    pred_mask.astype(np.float32),
                    masks[frame_idx].astype(np.float32)
                )

        video_result = {
            'video_idx': video_idx,
            'folder': folder_path,
            'num_frames': len(frame_names),
            'results': frame_results,
            'mean_iou': np.mean([r['iou'] for r in frame_results if r['has_target']])
        }

        if speed_info:
            video_result['speed'] = speed_info
            all_speed_info.append(speed_info)

        all_results.append(video_result)

    # 计算最终指标
    final_metrics = all_metrics.compute()

    # 计算Mean IoU
    all_ious = []
    for video_result in all_results:
        for frame_result in video_result['results']:
            if frame_result['has_target']:
                all_ious.append(frame_result['iou'])

    mean_iou = np.mean(all_ious) if all_ious else 0

    # 计算速度指标（benchmark模式）
    avg_fps = 0
    avg_frame_time = 0
    if args.benchmark and all_speed_info:
        total_frames = sum(s['num_frames'] for s in all_speed_info)
        total_time = sum(s['total_time_s'] for s in all_speed_info)
        avg_fps = total_frames / total_time if total_time > 0 else 0
        avg_frame_time = np.mean([s['avg_frame_time_ms'] for s in all_speed_info])

    # 打印结果
    print("\n" + "=" * 60)
    print("SAM 2 验证结果")
    print("=" * 60)
    print(f"视频数量: {len(all_results)}")
    print(f"总帧数: {len(all_ious)}")
    print(f"Mean IoU: {mean_iou:.4f}")

    if args.benchmark:
        print("-" * 60)
        print("性能指标:")
        print(f"平均FPS: {avg_fps:.2f}")
        print(f"平均推理时间: {avg_frame_time:.2f} ms/帧")
        print(f"总处理时间: {sum(s['total_time_s'] for s in all_speed_info):.2f} 秒")

    print("=" * 60)

    # 决策
    if mean_iou > 0.5:
        print("\n✅ SAM 2表现良好！可以进入阶段二（引导头训练）")
    elif mean_iou > 0.3:
        print("\n⚠️ SAM 2表现中等。考虑在阶段三进行LoRA微调")
    else:
        print("\n❌ SAM 2对小目标跟踪困难。需要LoRA微调！")

    # 保存结果报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'data_root': args.data_root,
            'test_list': args.test_list,
            'sam2_checkpoint': args.sam2_ckpt,
            'sam2_config': args.sam2_config,
            'target_label': args.target_label,
            'num_videos': len(all_results),
            'benchmark': args.benchmark
        },
        'metrics': {
            'mean_iou': float(mean_iou),
            'iou': float(final_metrics['iou']),
            'niou': float(final_metrics['niou']),
            'pd': float(final_metrics['pd']),
            'fa': float(final_metrics['fa'])
        },
        'per_video_results': [
            {
                'video_idx': r['video_idx'],
                'folder': r['folder'],
                'num_frames': r['num_frames'],
                'mean_iou': float(r['mean_iou']),
                'speed': r.get('speed')
            }
            for r in all_results
        ]
    }

    # 添加速度指标汇总
    if args.benchmark and all_speed_info:
        report['performance'] = {
            'avg_fps': float(avg_fps),
            'avg_frame_time_ms': float(avg_frame_time),
            'total_time_s': float(sum(s['total_time_s'] for s in all_speed_info)),
            'total_frames': sum(s['num_frames'] for s in all_speed_info)
        }

    with open(output_dir / 'validation_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到 {output_dir / 'validation_report.json'}")


if __name__ == '__main__':
    main()
