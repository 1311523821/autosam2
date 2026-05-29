#!/usr/bin/env python
"""
SAM2 独立测试 — 隔离测试微调后 SAM2 的视频目标分割能力。

与训练验证和 e2e 测试的区别：
  - 训练验证 (validate):    4帧clip + GT点 + 训练管线 → 监控训练进度
  - SAM2 独立测试 (本脚本):  完整视频 + GT点 + SAM2 predictor → 衡量 SAM2 本身的分割能力
  - E2E 测试 (evaluate_e2e): 完整视频 + Guidance Head + SAM2 + re-prompt → 衡量整套系统

为什么要独立测试 SAM2：
  微调 SAM2 的目标是提升其视频目标分割能力（给定提示点后，在视频中传播 mask 的精度）。
  如果直接用 e2e 测试，guidance head 的误差会掩盖 SAM2 的改进。
  只有当 SAM2 独立测试结果满意后，才应该进入 e2e 测试阶段。

用法:
    # 测试微调后的 SAM2
    python scripts/test_sam2_lora.py \
        --sam2-ckpt results/phase3_lora/.../checkpoints/best_inference.pth

    # 测试原始 SAM2（基线对比）
    python scripts/test_sam2_lora.py \
        --sam2-ckpt checkpoints/sam2.1_hiera_tiny.pt \
        --baseline

    # 对比模式（同时测微调+基线，输出对比报告）
    python scripts/test_sam2_lora.py \
        --sam2-ckpt results/phase3_lora/.../checkpoints/best_inference.pth \
        --compare-with checkpoints/sam2.1_hiera_tiny.pt

    # 启用 GT 重提示（模拟理想重提示场景）
    python scripts/test_sam2_lora.py --sam2-ckpt .../best_inference.pth --reprompt
"""

import os, sys, json, argparse, time, cv2
import torch, numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '/root/autosam2')

from utils.metrics import NUDT_Metrics
from utils.heatmap import get_mask_center


# ============================================================
# 视频数据加载
#   从原始视频文件夹加载帧 + GT mask + GT 中心点。
#   不做 resize（由 SAM2 predictor 内部处理）。
# ============================================================

def load_video(folder_path: str, target_label: str = 'uuv'):
    """
    加载完整视频。

    Returns:
        frames:    [(H, W, 3), ...] RGB uint8 帧列表（SAM2 predictor 期望的格式）
        gt_masks:  {frame_idx: (H, W) binary mask}
        gt_centers: [(x, y), ...] 每帧 GT 中心点（像素坐标），无目标帧为 None
        filenames: [str] 帧文件名
    """
    img_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    frames, gt_masks, gt_centers = [], {}, []

    for i, name in enumerate(img_files):
        img = cv2.imread(os.path.join(folder_path, name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        frames.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))

        # 加载 GT mask
        json_path = os.path.join(folder_path, os.path.splitext(name)[0] + '.json')
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for s in data.get('shapes', []):
                if s.get('label') == target_label:
                    pts = np.array(s['points'], dtype=np.float32).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)

        gt_masks[i] = mask
        center = get_mask_center(torch.from_numpy(mask))
        gt_centers.append((float(center[0]), float(center[1])) if center else None)

    return frames, gt_masks, gt_centers, img_files


# ============================================================
# SAM2 Predictor 初始化
#   baseline 模式：直接用官方 API 加载原始权重
#   微调模式：先构建 predictor 再 load_state_dict(strict=False)
#   因为 best_inference.pth 中的 LoRA 已合并，key 与原始模型一致
# ============================================================

def init_predictor(config_path: str, checkpoint_path: str, device: str,
                   is_baseline: bool = False):
    """初始化 SAM2 predictor"""
    from sam2.build_sam import build_sam2_video_predictor

    if is_baseline:
        # 原始 SAM2：直接通过官方 API 加载
        predictor = build_sam2_video_predictor(config_path, checkpoint_path, device=device)
        print(f"  原始 SAM2: {checkpoint_path}")
    else:
        # 微调 SAM2：先构建，再用 merged state_dict 覆盖
        ckpt = torch.load(checkpoint_path, map_location=device)
        if 'model' in ckpt:
            sd = ckpt['model']
        elif 'model_state_dict' in ckpt:
            sd = ckpt['model_state_dict']
        else:
            sd = ckpt

        predictor = build_sam2_video_predictor(config_path, device=device)
        missing, unexpected = predictor.load_state_dict(sd, strict=False)
        print(f"  微调 SAM2: {checkpoint_path}")
        if missing:
            print(f"    missing keys: {len(missing)} (预期)")
        if unexpected:
            print(f"    unexpected keys: {len(unexpected)}")

    return predictor


# ============================================================
# 视频传播 — GT 中心点驱动 SAM2 分割
#   给定 frame 0 的 GT 中心点，让 SAM2 在完整视频上传播 mask。
#   可选 GT 重提示：当检测到 mask 面积塌缩时，用下一帧的 GT 中心点重新注入。
# ============================================================

def run_video(predictor, frames, gt_centers, reprompt: bool = False,
              area_threshold: float = 0.3, distance_threshold: int = 10):
    """
    在单个视频上运行 SAM2 传播。

    Args:
        predictor: SAM2VideoPredictor
        frames: [(H, W, 3), ...] RGB 帧列表
        gt_centers: [(x, y), ...] 每帧 GT 中心点
        reprompt: 是否启用 GT 重提示
        area_threshold: 面积塌缩阈值（当前面积 < 上次面积 * threshold 时触发）
        distance_threshold: 距离阈值（新 GT 点与上次提示点的最小距离）

    Returns:
        {frame_idx: mask_array(H, W)}  每帧的预测 mask（二值）
    """
    # 初始化 SAM2 状态（帧列表方式，predictor 内部处理预处理）
    inference_state = predictor.init_state(
        video_path=None,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )
    # 手动设置帧（绕过视频路径加载）
    predictor.reset_state(inference_state)

    # 使用 frame 0 的 GT 中心点作为初始 prompt
    if gt_centers[0] is None:
        return {}
    pt0 = np.array([gt_centers[0]], dtype=np.float32)
    lbl0 = np.array([1], dtype=np.int32)

    # SAM2 内部 _get_image_feature 需要特定格式的帧存储
    # 直接使用内部 API 逐帧编码并存储
    for frame_idx, frame in enumerate(frames):
        # 存储原始帧（predictor 内部会在需要时编码）
        inference_state["images"].append(frame)
    inference_state["num_frames"] = len(frames)

    # 初始化第一帧
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=pt0,
        labels=lbl0,
    )

    # 传播
    pred_masks = {}
    prev_area = None
    last_prompt_pt = gt_centers[0]
    last_prompt_frame = 0

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        # predictor 输出是 raw logits (1, H, W)，需 sigmoid + squeeze + 二值化
        pred_logit = out_mask_logits[0].cpu().numpy()
        pred_prob = 1.0 / (1.0 + np.exp(-pred_logit))
        mask = (pred_prob > 0.5).astype(np.uint8).squeeze(0)
        pred_masks[out_frame_idx] = mask

        # GT 重提示逻辑
        if reprompt and out_frame_idx + 1 < len(frames):
            current_area = mask.sum()
            gt_next = gt_centers[out_frame_idx + 1]

            if gt_next is not None and prev_area is not None:
                # 双重验证：面积塌缩 + 距离变化
                area_collapse = current_area < prev_area * area_threshold
                dist = np.linalg.norm(np.array(gt_next) - np.array(last_prompt_pt))
                far_enough = dist > distance_threshold

                if area_collapse and far_enough:
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=out_frame_idx + 1,
                        obj_id=1,
                        points=np.array([gt_next], dtype=np.float32),
                        labels=np.array([1], dtype=np.int32),
                    )
                    last_prompt_pt = gt_next
                    last_prompt_frame = out_frame_idx + 1

            prev_area = current_area

    return pred_masks


# ============================================================
# 单视频评估（简化版，用于 baseline 比较时保持一致性）
#   直接使用 predictor.init_state 标准 API
# ============================================================

def run_video_simple(predictor, frames, gt_centers):
    """
    简化版：仅用 frame 0 GT 点初始化，然后传播。
    不重提示，与 baseline 测试方式一致。
    """
    # 使用标准 API：先将帧写入临时 numpy 数组
    video_np = np.stack(frames)  # (T, H, W, 3)

    inference_state = predictor.init_state(
        video=video_np,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )

    if gt_centers[0] is None:
        return {}

    pt0 = np.array([gt_centers[0]], dtype=np.float32)
    lbl0 = np.array([1], dtype=np.int32)

    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=pt0,
        labels=lbl0,
    )

    pred_masks = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        # predictor 输出是 raw logits (1, H, W)，需 sigmoid + squeeze + 二值化
        pred_logit = out_mask_logits[0].cpu().numpy()
        pred_prob = 1.0 / (1.0 + np.exp(-pred_logit))
        mask = (pred_prob > 0.5).astype(np.uint8).squeeze(0)
        pred_masks[out_frame_idx] = mask

    return pred_masks


# ============================================================
# 主测试逻辑
# ============================================================

def test_sam2(config_path, checkpoint_path, test_list, data_root, device,
              is_baseline=False, reprompt=False, area_threshold=0.3,
              distance_threshold=10, target_label='uuv'):
    """测试单个 SAM2 模型在测试集上的视频分割性能"""

    predictor = init_predictor(config_path, checkpoint_path, device, is_baseline)
    metrics = NUDT_Metrics(thre=0.5)

    with open(test_list, 'r') as f:
        video_folders = [l.strip().replace('\\', '/') for l in f if l.strip()]

    total_frames = 0
    total_time = 0
    video_results = []

    for folder in tqdm(video_folders, desc="测试视频"):
        video_path = os.path.join(data_root, folder)
        if not os.path.isdir(video_path):
            continue

        frames, gt_masks, gt_centers, _ = load_video(video_path, target_label)
        if len(frames) == 0:
            continue

        # 计时
        torch.cuda.synchronize()
        t0 = time.time()

        # 运行 SAM2 视频传播
        pred_masks = run_video_simple(predictor, frames, gt_centers)
        # TODO: 如需重提示，改用 run_video()

        torch.cuda.synchronize()
        elapsed = time.time() - t0

        # 计算指标
        video_iou_sum = 0
        video_frames = 0
        for fi, pred in pred_masks.items():
            if fi in gt_masks:
                metrics.update(pred, gt_masks[fi])
                video_iou_sum += (pred & gt_masks[fi]).sum() / max((pred | gt_masks[fi]).sum(), 1)
                video_frames += 1

        total_frames += len(frames)
        total_time += elapsed

        if video_frames > 0:
            video_results.append({
                'video': folder,
                'frames': video_frames,
                'video_iou': video_iou_sum / video_frames,
                'time_ms': elapsed * 1000 / len(frames),
            })

        # 清理
        del frames, gt_masks, pred_masks
        torch.cuda.empty_cache()

    iou, niou, pd, fa = metrics.get()
    fps = total_frames / total_time if total_time > 0 else 0

    return {
        'IoU': iou, 'nIoU': niou, 'Pd': pd, 'Fa': fa,
        'FPS': fps,
        'total_frames': total_frames,
        'total_videos': len(video_results),
        'video_results': video_results,
    }


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description='SAM2 独立测试（GT 点 + 视频传播）')

    p.add_argument('--sam2-config', default='sam2_hiera_t.yaml')
    p.add_argument('--sam2-ckpt', required=True,
                   help='SAM2 权重路径（best_inference.pth 或原始 checkpoint）')
    p.add_argument('--data-root', default='/root/DataBscan')
    p.add_argument('--test-list', default='/root/e2e/test1.txt')
    p.add_argument('--output', default=None,
                   help='输出 JSON 路径（默认自动生成到 results/）')
    p.add_argument('--baseline', action='store_true',
                   help='作为基线模型测试（使用官方 API 直接加载原始权重）')
    p.add_argument('--compare-with', default=None,
                   help='同时测试另一个模型并输出对比报告')
    p.add_argument('--reprompt', action='store_true',
                   help='启用 GT 重提示')
    p.add_argument('--area-threshold', type=float, default=0.3)
    p.add_argument('--distance-threshold', type=int, default=10)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("SAM2 独立测试（GT 点 → 视频传播）")
    print("=" * 60)
    print(f"  模型: {args.sam2_ckpt}")
    print(f"  模式: {'基线' if args.baseline else '微调'}")
    print(f"  设备: {args.device}")

    # 测试主模型
    results = test_sam2(
        args.sam2_config, args.sam2_ckpt, args.test_list,
        args.data_root, args.device,
        is_baseline=args.baseline,
        reprompt=args.reprompt,
        area_threshold=args.area_threshold,
        distance_threshold=args.distance_threshold,
    )

    print(f"\n{'=' * 60}")
    print("测试结果")
    print(f"{'=' * 60}")
    print(f"  IoU:  {results['IoU']:.4f}")
    print(f"  nIoU: {results['nIoU']:.4f}")
    print(f"  Pd:   {results['Pd']:.3f}")
    print(f"  Fa:   {results['Fa']:.5f}")
    print(f"  FPS:  {results['FPS']:.1f}")
    print(f"  帧数: {results['total_frames']}")

    # 对比模式
    if args.compare_with:
        print(f"\n{'=' * 60}")
        print("对比基线")
        print(f"{'=' * 60}")
        baseline_results = test_sam2(
            args.sam2_config, args.compare_with, args.test_list,
            args.data_root, args.device,
            is_baseline=True,
            reprompt=args.reprompt,
        )

        print(f"\n  指标        基线        微调        变化")
        print(f"  {'─' * 45}")
        for key, label in [('IoU', 'IoU'), ('nIoU', 'nIoU'), ('Pd', 'Pd'), ('Fa', 'Fa')]:
            b = baseline_results[key]
            f = results[key]
            delta = f - b
            direction = '↑' if delta > 0 else '↓'
            print(f"  {label:8s}  {b:.4f}     {f:.4f}     {delta:+.4f} {direction}")

        # 保存对比报告
        results['baseline'] = baseline_results
        results['comparison'] = {
            'IoU_delta': results['IoU'] - baseline_results['IoU'],
            'nIoU_delta': results['nIoU'] - baseline_results['nIoU'],
            'Pd_delta': results['Pd'] - baseline_results['Pd'],
            'Fa_delta': results['Fa'] - baseline_results['Fa'],
        }

    # 保存
    if args.output is None:
        ckpt_name = Path(args.sam2_ckpt).stem
        tag = 'baseline' if args.baseline else 'finetuned'
        args.output = f'results/sam2_test_{tag}_{ckpt_name}.json'

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.',
                exist_ok=True)
    # 转换 numpy 为 Python 原生类型
    results_serializable = {
        k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
        for k, v in results.items()
        if k != 'video_results'  # 保持嵌套结构
    }
    results_serializable['video_results'] = results.get('video_results', [])
    if 'baseline' in results:
        results_serializable['baseline'] = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in results['baseline'].items() if k != 'video_results'
        }
        results_serializable['comparison'] = {
            k: float(v) for k, v in results['comparison'].items()
        }

    with open(args.output, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\n结果已保存: {args.output}")


if __name__ == '__main__':
    main()
