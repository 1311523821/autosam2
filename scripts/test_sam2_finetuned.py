#!/usr/bin/env python
"""
微调后SAM 2测试脚本

使用GT中心点测试SAM 2性能，隔离引导头误差。

用法:
    # 测试微调后的模型
    python scripts/test_sam2_finetuned.py \
        --sam2-ckpt results/phase3_sam2_finetune/checkpoints/best.pth \
        --test-list /root/e2e/test1.txt

    # 测试原始冻结模型（对比基线）
    python scripts/test_sam2_finetuned.py \
        --sam2-ckpt checkpoints/sam2.1_hiera_tiny.pt \
        --test-list /root/e2e/test1.txt \
        --baseline
"""

import os
import sys
import argparse
import json
import cv2
import torch
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '/root/autosam2')
sys.path.insert(0, '/root/e2e')

from utils.metrics import NUDT_Metrics, compute_iou
from utils.heatmap import get_mask_center


def parse_args():
    parser = argparse.ArgumentParser(description='测试微调后的SAM 2')
    parser.add_argument('--sam2-config', type=str,
                        default='configs/sam2.1/sam2.1_hiera_t.yaml',
                        help='SAM 2配置文件')
    parser.add_argument('--sam2-ckpt', type=str, required=True,
                        help='SAM 2权重路径（微调后或原始）')
    parser.add_argument('--data-root', type=str, default='/root/DataBscan',
                        help='数据根目录')
    parser.add_argument('--test-list', type=str, default='/root/e2e/test1.txt',
                        help='测试列表文件')
    parser.add_argument('--output', type=str, default='results/phase3_sam2_test',
                        help='输出目录')
    parser.add_argument('--target-label', type=str, default='uuv',
                        help='目标标签')
    parser.add_argument('--visualize', action='store_true',
                        help='保存可视化结果')
    parser.add_argument('--benchmark', action='store_true',
                        help='启用FPS计时')
    parser.add_argument('--baseline', action='store_true',
                        help='测试原始冻结模型（基线对比）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='运行设备')
    return parser.parse_args()


def load_video_data(folder_path: str, target_label: str):
    """加载视频帧和GT数据"""
    valid_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    frames, masks, centers = [], [], []

    for name in valid_files:
        # 加载帧
        img = cv2.imread(os.path.join(folder_path, name), cv2.IMREAD_GRAYSCALE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        frames.append(img_rgb)

        # 加载GT mask
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
                print(f"加载 {json_path} 错误: {e}")

        masks.append(mask)

        # 计算中心点
        center = get_mask_center(torch.from_numpy(mask))
        centers.append((center[0], center[1]) if center is not None else None)

    return frames, masks, centers, valid_files


def init_predictor(config_path: str, checkpoint_path: str, device: str, is_baseline: bool = False):
    """初始化SAM 2预测器"""
    try:
        from sam2.build_sam import build_sam2_video_predictor

        # 微调后的 best_inference.pth 使用 checkpoint["model"] 格式，
        # build_sam2_video_predictor 内部直接加载，无需手动 load_state_dict
        predictor = build_sam2_video_predictor(config_path, checkpoint_path, device=device)

        if is_baseline:
            print(f"加载原始SAM 2模型: {checkpoint_path}")
        else:
            print(f"加载微调后的SAM 2模型: {checkpoint_path}")
            print("✓ 加载微调权重成功")

        return predictor
    except Exception as e:
        print(f"加载SAM 2错误: {e}")
        return None


def test_video(
    predictor,
    folder_path: str,
    gt_masks: list,
    gt_centers: list,
    video_idx: int,
    output_dir: str = None,
    visualize: bool = False,
    benchmark: bool = False
):
    """
    测试单个视频

    使用GT中心点作为prompt，测试SAM 2性能。
    """
    # 找第一个有效中心点
    prompt_idx = None
    prompt_point = None
    for i, center in enumerate(gt_centers):
        if center is not None:
            prompt_idx = i
            prompt_point = np.array([[center[0], center[1]]])
            break

    if prompt_point is None:
        return None

    # 计时
    init_start = time.time()
    inference_state = predictor.init_state(
        folder_path,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True
    )
    init_time = time.time() - init_start

    # 注入Prompt
    predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=prompt_idx,
        obj_id=1,
        points=prompt_point,
        labels=np.array([1], dtype=np.int32)
    )

    # 视频传播
    results = []
    video_segments = {}
    total_time = 0.0
    frame_times = []

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        frame_start = time.time()

        mask_logits = out_mask_logits[0]  # (1, H, W)
        pred_mask = (mask_logits > 0).cpu().numpy().squeeze()

        gt_mask = gt_masks[out_frame_idx] if out_frame_idx < len(gt_masks) else None

        if gt_mask is not None:
            iou = compute_iou(
                torch.from_numpy(pred_mask).float(),
                torch.from_numpy(gt_mask).float()
            )
        else:
            iou = 0.0

        frame_time = time.time() - frame_start
        if benchmark:
            frame_times.append(frame_time)
            total_time += frame_time

        results.append({
            'frame_idx': out_frame_idx,
            'iou': iou,
            'has_target': gt_mask is not None and gt_mask.sum() > 0
        })

        video_segments[out_frame_idx] = pred_mask

    # 可视化
    if visualize and output_dir:
        vis_dir = os.path.join(output_dir, 'visualizations', f'video_{video_idx}')
        os.makedirs(vis_dir, exist_ok=True)

        frame_files = sorted([f for f in os.listdir(folder_path)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        for frame_idx in range(len(frame_files)):
            if frame_idx not in video_segments:
                continue

            frame_path = os.path.join(folder_path, frame_files[frame_idx])
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            pred_mask = video_segments[frame_idx]
            gt_mask = gt_masks[frame_idx] if frame_idx < len(gt_masks) else None

            # 绘制GT mask（绿色）
            if gt_mask is not None:
                gt_overlay = np.zeros_like(frame)
                gt_overlay[gt_mask > 0] = [0, 255, 0]
                frame = cv2.addWeighted(frame, 0.7, gt_overlay, 0.3, 0)

            # 绘制预测mask（红色）
            pred_overlay = np.zeros_like(frame)
            pred_overlay[pred_mask > 0] = [0, 0, 255]
            frame = cv2.addWeighted(frame, 0.7, pred_overlay, 0.3, 0)

            # 绘制中心点
            if frame_idx < len(gt_centers) and gt_centers[frame_idx] is not None:
                cx, cy = gt_centers[frame_idx]
                cv2.circle(frame, (int(cx), int(cy)), 5, (255, 255, 0), -1)

            # 添加信息
            iou = results[frame_idx]['iou'] if frame_idx < len(results) else 0
            cv2.putText(frame, f'Frame {frame_idx}, IoU: {iou:.3f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imwrite(os.path.join(vis_dir, f'frame_{frame_idx:04d}.png'), frame)

    # 清理
    del inference_state
    torch.cuda.empty_cache()

    return {
        'results': results,
        'video_segments': video_segments,
        'speed': {
            'init_time_s': init_time,
            'total_time_s': total_time,
            'num_frames': len(frame_times),
            'avg_frame_time_ms': np.mean(frame_times) * 1000 if frame_times else 0,
            'fps': len(frame_times) / total_time if total_time > 0 else 0
        } if benchmark else None
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("SAM 2 微调测试" if not args.baseline else "SAM 2 基线测试")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载测试列表
    with open(args.test_list, 'r') as f:
        test_folders = [line.strip().replace('\\', '/') for line in f if line.strip()]

    print(f"测试视频: {len(test_folders)}")
    print(f"Benchmark模式: {'开启' if args.benchmark else '关闭'}")

    # 初始化预测器
    predictor = init_predictor(
        args.sam2_config, args.sam2_ckpt, args.device, args.baseline
    )
    if predictor is None:
        print("预测器初始化失败")
        return

    # 指标
    metrics = NUDT_Metrics()
    all_results = []
    all_speed_info = []

    # 测试每个视频
    for video_idx, folder in enumerate(tqdm(test_folders, desc="测试中")):
        folder_path = os.path.join(args.data_root, folder)
        if not os.path.isdir(folder_path):
            continue

        # 加载数据
        _, gt_masks, gt_centers, frame_files = load_video_data(folder_path, args.target_label)

        # 测试
        result = test_video(
            predictor, folder_path, gt_masks, gt_centers,
            video_idx, str(output_dir), args.visualize, args.benchmark
        )

        if result is None:
            continue

        frame_results = result['results']
        video_segments = result['video_segments']
        speed_info = result['speed']

        # 聚合指标
        for frame_idx, pred_mask in video_segments.items():
            if frame_idx < len(gt_masks):
                metrics.update(
                    pred_mask.astype(np.float32),
                    gt_masks[frame_idx].astype(np.float32)
                )

        # 记录结果
        video_result = {
            'video_idx': video_idx,
            'folder': folder,
            'num_frames': len(frame_files),
            'mean_iou': np.mean([r['iou'] for r in frame_results if r['has_target']]),
        }

        if speed_info:
            video_result['speed'] = speed_info
            all_speed_info.append(speed_info)

        all_results.append(video_result)

    # 计算最终指标
    final_metrics = metrics.compute()

    # 计算Mean IoU
    all_ious = []
    for video_result in all_results:
        for r in video_result.get('results', []):
            if isinstance(r, dict) and r.get('has_target'):
                all_ious.append(r['iou'])

    mean_iou = np.mean(all_ious) if all_ious else 0

    # 计算速度
    avg_fps = 0
    if args.benchmark and all_speed_info:
        total_frames = sum(s['num_frames'] for s in all_speed_info)
        total_time = sum(s['total_time_s'] for s in all_speed_info)
        avg_fps = total_frames / total_time if total_time > 0 else 0

    # 打印结果
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    print(f"视频数量: {len(all_results)}")
    print(f"总帧数: {len(all_ious)}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"IoU (累积): {final_metrics['iou']:.4f}")
    print(f"nIoU: {final_metrics['niou']:.4f}")
    print(f"Pd: {final_metrics['pd']:.4f}")
    print(f"Fa: {final_metrics['fa']:.4f}")

    if args.benchmark:
        print("-" * 60)
        print(f"平均FPS: {avg_fps:.2f}")

    print("=" * 60)

    # 保存报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'sam2_checkpoint': args.sam2_ckpt,
            'sam2_config': args.sam2_config,
            'data_root': args.data_root,
            'test_list': args.test_list,
            'is_baseline': args.baseline,
            'num_videos': len(all_results)
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

    if args.benchmark:
        report['performance'] = {
            'avg_fps': float(avg_fps)
        }

    report_path = output_dir / 'test_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n结果已保存到: {report_path}")


if __name__ == '__main__':
    main()
