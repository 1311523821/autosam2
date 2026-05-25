"""
端到端评估脚本

评估 AutoPromptSAM2 在测试集上的性能。
输出指标：IoU, nIoU, Pd, Fa, FPS, 推理时间, 权重大小
"""

import os
import sys
import json
import time
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, '/root/autosam2')
sys.path.insert(0, '/root/e2e')

from models.auto_prompt_sam2 import build_auto_prompt_sam2
from utils.metrics import NUDT_Metrics


def load_video_data(video_folder: str, target_label: str = 'uuv'):
    """
    加载视频帧和GT标注

    Returns:
        frames: (T, H, W) 灰度帧
        gt_masks: {frame_idx: mask}
    """
    # 获取图片文件
    img_files = sorted([
        f for f in os.listdir(video_folder)
        if f.endswith(('.png', '.jpg', '.bmp'))
    ])

    # 加载灰度帧
    frames = []
    for img_file in img_files:
        img_path = os.path.join(video_folder, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            frames.append(img)

    # 加载每帧单独的JSON标注
    gt_masks = {}

    for i, img_file in enumerate(img_files):
        # 查找对应的JSON文件
        json_file = img_file.rsplit('.', 1)[0] + '.json'
        json_path = os.path.join(video_folder, json_file)

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                annotation = json.load(f)

            mask = np.zeros(frames[0].shape[:2], dtype=np.uint8)

            for shape in annotation.get('shapes', []):
                if shape.get('label') == target_label:
                    points = np.array(shape['points'], dtype=np.int32)
                    cv2.fillPoly(mask, [points], 1)

            gt_masks[i] = mask

    return frames, gt_masks


def get_model_size_mb(model):
    """计算模型大小（MB）"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def evaluate_e2e(
    guidance_ckpt: str,
    sam2_ckpt: str,
    sam2_config: str,
    test_list: str,
    data_root: str,
    output: str,
    area_threshold: float = 0.3,
    distance_threshold: int = 10,
    target_label: str = 'uuv'
):
    """
    端到端评估
    """
    # 构建模型
    print("=" * 60)
    print("AutoPromptSAM2 端到端评估")
    print("=" * 60)

    guidance_config = {
        'in_chans': 1,
        'embed_dim': 64,
        'hidden_dim': 32,
        'num_frame': 4
    }

    model = build_auto_prompt_sam2(
        guidance_config=guidance_config,
        guidance_checkpoint=guidance_ckpt,  # 传递引导头权重路径
        sam2_checkpoint=sam2_ckpt,
        sam2_config=sam2_config,
        area_threshold=area_threshold,
        distance_threshold=distance_threshold,
        device='cuda'
    )

    # 计算模型大小
    model_size = get_model_size_mb(model)
    print(f"模型大小: {model_size:.2f} MB")

    # 读取测试列表
    with open(test_list, 'r') as f:
        video_folders = [line.strip() for line in f if line.strip()]

    print(f"测试视频数: {len(video_folders)}")

    # 初始化指标
    metrics = NUDT_Metrics(thre=0.5)

    # 计时
    total_frames = 0
    total_time = 0
    total_reprompts = 0

    # 遍历视频
    for video_folder in tqdm(video_folders, desc="评估视频"):
        # 修复Windows路径分隔符
        video_folder = video_folder.replace('\\', '/')
        video_path = os.path.join(data_root, video_folder)

        if not os.path.exists(video_path):
            print(f"警告: 视频不存在 {video_path}")
            continue

        # 加载GT标注
        frames, gt_masks = load_video_data(video_path, target_label)

        if len(frames) == 0:
            continue

        T = len(frames)

        # 计时开始
        torch.cuda.synchronize()
        start_time = time.time()

        # 处理视频（传入文件夹路径）
        pred_masks, reprompt_count = model.process_video_with_reprompt(video_path)

        # 计时结束
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time

        # 累积指标
        for frame_idx, pred_mask in pred_masks.items():
            if frame_idx in gt_masks:
                pred_np = pred_mask.cpu().numpy().squeeze().astype(np.uint8)
                gt_np = gt_masks[frame_idx]
                metrics.update(pred_np, gt_np)

        total_frames += T
        total_time += elapsed_time
        total_reprompts += reprompt_count

        # 清理GPU内存
        del pred_masks
        del frames
        del gt_masks
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # 计算最终指标
    iou, niou, pd, fa = metrics.get()

    # 构建结果
    results = {
        'IoU': float(iou),
        'nIoU': float(niou),
        'Pd': float(pd),
        'Fa': float(fa),
        'FPS': float(total_frames / total_time) if total_time > 0 else 0,
        'inference_time_ms': float(total_time / total_frames * 1000) if total_frames > 0 else 0,
        'model_size_mb': float(model_size),
        'total_frames': total_frames,
        'total_videos': len(video_folders),
        'total_reprompts': total_reprompts,
        'config': {
            'guidance_ckpt': guidance_ckpt,
            'sam2_ckpt': sam2_ckpt,
            'sam2_config': sam2_config,
            'area_threshold': area_threshold,
            'distance_threshold': distance_threshold
        }
    }

    # 输出结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"IoU:              {results['IoU']:.4f}")
    print(f"nIoU:             {results['nIoU']:.4f}")
    print(f"Pd:               {results['Pd']:.4f}")
    print(f"Fa:               {results['Fa']:.6f}")
    print(f"FPS:              {results['FPS']:.2f}")
    print(f"推理时间:         {results['inference_time_ms']:.2f} ms/帧")
    print(f"模型大小:         {results['model_size_mb']:.2f} MB")
    print(f"总帧数:           {results['total_frames']}")
    print(f"Re-prompt次数:    {results['total_reprompts']}")
    print("=" * 60)

    # 保存结果
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else '.', exist_ok=True)
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存到: {output}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoPromptSAM2 端到端评估')
    # 使用正确的checkpoint路径
    parser.add_argument('--guidance-ckpt', type=str,
                        default=r'checkpoints/experiments/exp6_embed64/enc(stsf)_embed(64)_hidden(32)_opt(adamw)_lr(5e-4)_loss(focal_a2)_sigma(10)_kfold(5)/fold_1/best.pth',
                        help='引导头权重路径')
    parser.add_argument('--sam2-ckpt', type=str, default='checkpoints/sam2.1_hiera_tiny.pt', help='SAM2权重路径')
    parser.add_argument('--sam2-config', type=str, default='configs/sam2.1/sam2.1_hiera_t.yaml',
                        help='SAM2配置文件')
    parser.add_argument('--test-list', type=str, default='/root/e2e/test1.txt', help='测试列表文件')
    parser.add_argument('--data-root', type=str, default='/root/DataBscan', help='数据根目录')
    parser.add_argument('--output', type=str, default='results/e2e_tiny.json',
                        help='输出文件路径')
    parser.add_argument('--area-threshold', type=float, default=0.3,
                        help='面积突变阈值')
    parser.add_argument('--distance-threshold', type=int, default=10,
                        help='距离验证阈值（像素）')
    parser.add_argument('--target-label', type=str, default='uuv',
                        help='目标标签')

    args = parser.parse_args()

    evaluate_e2e(
        guidance_ckpt=args.guidance_ckpt,
        sam2_ckpt=args.sam2_ckpt,
        sam2_config=args.sam2_config,
        test_list=args.test_list,
        data_root=args.data_root,
        output=args.output,
        area_threshold=args.area_threshold,
        distance_threshold=args.distance_threshold,
        target_label=args.target_label
    )
