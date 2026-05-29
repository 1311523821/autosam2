#!/usr/bin/env python
"""
SAM2 LoRA 训练脚本

通过配置切换 LoRA 注入目标（cross_attn / self_attn / memory_attn / image_encoder），
支持梯度累积、混合精度、双优化器、warmup+cosine 学习率调度、早停。

训练流水线（每 4 帧 clip）:
  Frame 0: GT 中心点 → image_encoder → mask_decoder → loss_0
            └→ memory_encoder → memory_0
  Frame 1-3: 上一帧预测 mask 的 argmax 点 → image_encoder → memory_attention(memories)
              → mask_decoder → loss (点是近似值，cross-attn 必须修正位置)
              └→ memory_encoder → 累积 memory

验证流水线（与测试一致）:
  非重叠 4 帧 clip → 完整 memory attention 管线 → NUDT 指标（IoU/nIoU/Pd/Fa）

用法:
    # 默认：cross-attention LoRA
    python scripts/train_sam2_lora.py

    # 切换目标 + 秩 + 优化器
    python scripts/train_sam2_lora.py --lora-targets memory_attn --lora-rank 8 --optimizer muon_adam

    # 冻结 mask_decoder（只用 LoRA 微调 cross-attn）
    python scripts/train_sam2_lora.py --no-mask-decoder

    # 手动指定输出目录（不自动生成）
    python scripts/train_sam2_lora.py --output results/my_experiment

    # 恢复训练
    python scripts/train_sam2_lora.py --resume results/.../checkpoints/epoch_20.pth
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/root/autosam2')

from models.sam2_lora_trainer import (
    SAM2LoRATrainer, CombinedLoss, build_trainer,
)
from utils.heatmap import get_mask_center


# ============================================================
# SonarClipDataset — 视频片段数据集
#
#   从声纳视频中提取连续 4 帧片段，用于 Memory Attention 训练。
#   训练时使用 50% 重叠滑动窗口（stride=clip_len//2）以增加样本量，
#   验证时使用非重叠窗口（stride=clip_len）以避免重复统计。
#
#   数据格式：
#   - 输入: 灰度 PNG/JPG，cv2 读取后 resize 到 image_size×image_size，
#           转为 3 通道并做 ImageNet 归一化（适配 SAM2 的 RGB 预训练权重）
#   - 标注: LabelMe JSON 格式，polygon → 二值 mask → 质心 → 缩放后的像素坐标
#   - 输出: (T, 3, H, W) 图像, (T, 2) 中心点, (T, H, W) mask
# ============================================================

class SonarClipDataset(Dataset):
    """
    4 帧 clip 数据集。

    预处理流水线（每帧）:
      cv2.imread(GRAYSCALE) → resize(1024,1024) → /255 → expand(3通道)
      → (x - mean) / std  (ImageNet 归一化)

    标注处理:
      JSON shapes[].points → cv2.fillPoly → 二值 mask
      → get_mask_center(质心) → 坐标缩放 (orig → 1024)
    """

    IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, folders, data_root, target_label='uuv',
                 image_size=1024, clip_len=4):
        self.data_root = data_root
        self.target_label = target_label
        self.image_size = image_size
        self.clip_len = clip_len
        self.clips = []

        import cv2

        for folder in tqdm(folders, desc="构建 clip 数据集"):
            d = os.path.join(data_root, folder)
            if not os.path.isdir(d):
                continue
            # 只取有对应 JSON 标注的帧
            files = sorted([
                f for f in os.listdir(d)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            valid = [
                f for f in files
                if os.path.exists(os.path.join(d, os.path.splitext(f)[0] + '.json'))
            ]
            # 50% 重叠滑动窗口，增加训练样本量
            for i in range(0, len(valid) - clip_len + 1, clip_len // 2):
                self.clips.append((d, valid[i:i + clip_len]))

        # 预过滤：加载失败或无目标的 clip 直接丢弃
        valid_clips = []
        for folder, files in self.clips:
            ok = True
            for f in files:
                try:
                    self._load_sample(folder, f)
                except Exception:
                    ok = False
                    break
            if ok:
                valid_clips.append((folder, files))
        self.clips = valid_clips

    def _load_sample(self, folder, img_file):
        """
        加载单帧 → 图像 tensor + 中心点 + mask。

        Raises:
            ValueError: 图像无法读取或无标注目标
        """
        import cv2

        img = cv2.imread(os.path.join(folder, img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取 {img_file}")
        h, w = img.shape

        # 读取 polygon 标注并填充 mask
        json_path = os.path.join(folder, os.path.splitext(img_file)[0] + '.json')
        mask = np.zeros((h, w), dtype=np.uint8)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for s in data.get('shapes', []):
            if s['label'] == self.target_label:
                pts = np.array(s['points'], dtype=np.float32).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)

        # 计算 mask 质心作为 prompt 点
        center = get_mask_center(torch.from_numpy(mask))
        if center is None or mask.sum() == 0:
            raise ValueError("无目标")

        # 缩放坐标到 image_size×image_size
        sx, sy = self.image_size / w, self.image_size / h

        # Resize 图像和 mask
        img_r = cv2.resize(img, (self.image_size, self.image_size))
        mask_r = cv2.resize(mask, (self.image_size, self.image_size),
                            interpolation=cv2.INTER_NEAREST)

        # 灰度 → 3 通道 → ImageNet 归一化（匹配 SAM2 预训练分布）
        img_t = torch.from_numpy(img_r).float() / 255.0
        img_t = img_t.unsqueeze(0).expand(3, -1, -1)
        img_t = (img_t - self.IMG_MEAN) / self.IMG_STD

        return img_t, np.array([center[0] * sx, center[1] * sy], dtype=np.float32), \
            torch.from_numpy(mask_r).float()

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        folder, files = self.clips[idx]
        images, points, masks = [], [], []
        for f in files:
            img_t, pt, mk = self._load_sample(folder, f)
            images.append(img_t)
            points.append(pt)
            masks.append(mk)
        return (
            torch.stack(images),                  # (T, 3, H, W)
            torch.from_numpy(np.stack(points)),   # (T, 2)
            torch.stack(masks),                   # (T, H, W)
        )


# ============================================================
# 验证 — 官方 predictor + 仅 frame 0 GT 点 → clip 级传播
#
#   验证逻辑与测试 (test_sam2_lora.py) 完全对齐：
#   - 仅首帧给 GT 中心点，后续帧模型自主传播
#   - 使用官方 SAM2 predictor（含 7 帧 memory bank 管理）
#   - 指标：NUDT IoU/nIoU/Pd/Fa
#
#   与训练 forward 的区别：
#   - 训练 _forward_clip: 帧0给GT点 + 帧1+用prev_mask argmax点 + 手写memory attention
#   - 验证 validate:     仅首帧 GT 点 + 官方 predictor（测真实传播能力）
#
#   为什么用 clip 而非完整视频做验证：
#   - 8 帧 clip 已足够评估短期传播能力
#   - 速度快，每 N epoch 跑一次不拖慢训练
#   - 完整视频测试由 test_sam2_lora.py 负责
# ============================================================

def _build_val_predictor(trainer, device):
    """
    从训练中的 LoRA 模型构建一个临时 predictor（用于验证传播能力）。

    SAM2VideoPredictor 继承自 SAM2Base，本身就是模型——没有独立的 .model 属性。
    必须通过 load_state_dict 注入合并后的 LoRA 权重。
    验证用 torch.no_grad()，predictor 的 @torch.inference_mode() 不影响。
    """
    import logging
    from sam2.build_sam import build_sam2_video_predictor

    # 抑制 Hydra/OmegaConf 初始化日志（每次验证都会重新构建 predictor）
    saved = {}
    for name in ["hydra", "omegaconf"]:
        logger = logging.getLogger(name)
        saved[name] = logger.level
        logger.setLevel(logging.WARNING)

    try:
        # 构建空 predictor（不加载 checkpoint）
        predictor = build_sam2_video_predictor(
            trainer.sam2_config, device=device
        )
    finally:
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)
    # 合并 LoRA 权重并通过 load_state_dict 注入（与 test_sam2_lora.py 同一机制）
    merged_sd = trainer._build_merged_state_dict()
    missing, unexpected = predictor.load_state_dict(merged_sd, strict=False)
    if missing or unexpected:
        print(f"  [val predictor] missing={len(missing)}, unexpected={len(unexpected)}")
    return predictor


def validate(trainer, test_folders, data_root, image_size, clip_len,
             loss_fn, device, sam2_config):
    """
    验证：官方 predictor + 仅 frame 0 GT 点 + clip 级传播 + NUDT 指标。

    与训练时的 _forward_clip 不同：
    - 不手写 memory pipeline，走官方 predictor.propagate_in_video()
    - 仅首帧注入 GT 点，后续帧完全自主传播
    - 测的是 SAM2 的真实视频追踪能力（而非每帧分割精度）

    Args:
        trainer: SAM2LoRATrainer（含当前 LoRA 权重）
        test_folders: 验证视频文件夹列表
        data_root: 数据根目录
        image_size: 图像 resize 尺寸
        clip_len: 验证 clip 帧数
        loss_fn: 损失函数
        device: 设备
        sam2_config: SAM2 配置文件路径

    Returns:
        {'loss', 'iou', 'niou', 'dice', 'pd', 'fa'}
    """
    from utils.metrics import NUDT_Metrics

    trainer.eval()
    predictor = _build_val_predictor(trainer, device)
    metrics = NUDT_Metrics(thre=0.5)
    total_loss = 0.0
    total_frames = 0

    # 为每个视频构建 clip
    all_clips = []
    for folder in test_folders:
        d = os.path.join(data_root, folder)
        if not os.path.isdir(d):
            continue
        files = sorted([f for f in os.listdir(d)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        valid = [f for f in files
                 if os.path.exists(os.path.join(d, os.path.splitext(f)[0] + '.json'))]
        # 非重叠 clip
        for i in range(0, len(valid) - clip_len + 1, clip_len):
            all_clips.append((d, valid[i:i + clip_len]))

    if len(all_clips) == 0:
        trainer.train()
        return {'loss': 0, 'iou': 0, 'niou': 0, 'dice': 0, 'pd': 0, 'fa': 0}

    with torch.no_grad():
        for folder, img_files in tqdm(all_clips, desc="验证"):
            # --- 加载原始帧（不做预处理，由 predictor 内部处理） ---
            raw_frames = []
            gt_masks_for_clip = {}
            gt_center_frame0 = None

            for fi, fname in enumerate(img_files):
                img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    break
                # predictor 期望 RGB uint8
                raw_frames.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))

                # GT mask（原始分辨率，与 predictor 输出对齐）
                json_path = os.path.join(folder, os.path.splitext(fname)[0] + '.json')
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    for s in data.get('shapes', []):
                        if s.get('label') == 'uuv':
                            pts = np.array(s['points'], dtype=np.float32).astype(np.int32)
                            cv2.fillPoly(mask, [pts], 1)

                # 首帧 GT 中心点
                if fi == 0 and mask.sum() > 0:
                    center = get_mask_center(torch.from_numpy(mask))
                    if center is not None:
                        gt_center_frame0 = (float(center[0]), float(center[1]))

                gt_masks_for_clip[fi] = mask

            if len(raw_frames) < clip_len or gt_center_frame0 is None:
                continue

            # --- 官方 predictor 传播 ---
            # 手动构造 inference_state（等同于 init_state 但接受内存帧）
            # SAM2 期望方形输入，需要 resize + ImageNet 归一化
            img_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
            img_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
            images_tensors = []
            for f in raw_frames:
                # resize 到方形（与 SAM2 image_size 匹配）
                f_resized = cv2.resize(f, (image_size, image_size))
                t = torch.from_numpy(f_resized).permute(2, 0, 1).float().to(device) / 255.0
                t = (t - img_mean) / img_std
                images_tensors.append(t)

            inference_state = {
                "images": images_tensors,
                "num_frames": len(images_tensors),
                "video_height": raw_frames[0].shape[0],
                "video_width": raw_frames[0].shape[1],
                "device": device,
                "storage_device": torch.device('cpu'),
                "point_inputs_per_obj": {},
                "mask_inputs_per_obj": {},
                "cached_features": {},
                "constants": {},
                "obj_id_to_idx": {},
                "obj_idx_to_id": {},
                "obj_ids": [],
                "output_dict_per_obj": {},
                "temp_output_dict_per_obj": {},
                "frames_tracked_per_obj": {},
                "offload_video_to_cpu": True,
                "offload_state_to_cpu": True,
            }

            pt0 = np.array([gt_center_frame0], dtype=np.float32)
            lbl0 = np.array([1], dtype=np.int32)

            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=pt0,
                labels=lbl0,
            )

            # 传播并收集预测
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state
            ):
                if out_frame_idx in gt_masks_for_clip:
                    gt = gt_masks_for_clip[out_frame_idx]
                    if gt.sum() == 0:
                        continue

                    # predictor 输出是 raw logits (1, H, W)，需要 sigmoid + squeeze + 二值化
                    pred_logit = out_mask_logits[0].cpu().numpy()
                    pred_prob = 1.0 / (1.0 + np.exp(-pred_logit))  # sigmoid
                    pred_mask = (pred_prob > 0.5).astype(np.uint8).squeeze(0)  # (1,H,W) → (H,W)
                    metrics.update(pred_mask, gt)
                    total_frames += 1

            # 清理
            del inference_state, images_tensors, raw_frames
            torch.cuda.empty_cache()

    trainer.train()

    n = max(total_frames, 1)
    iou, niou, pd, fa = metrics.get()

    # 用 -nIoU 作为 best_model 选择的 proxy"loss"（nIoU 越高越好，负数使"越小越好"的比较逻辑成立）
    return {
        'loss': 1.0 - niou,
        'iou': iou,
        'niou': niou,
        'dice': 2 * iou / (1 + iou) if iou < 1 else 1.0,
        'pd': pd,
        'fa': fa,
    }


# ============================================================
# 学习率调度 — Warmup + Cosine Annealing
#
#   Warmup 阶段（前 warmup_epochs 轮）：线性从 0 增长到 base_lr
#   Cosine 阶段：cosine 衰减从 base_lr 到 min_lr_ratio * base_lr
#
#   为什么用 warmup：
#     LoRA 参数从零初始化（B=0），初始梯度较大，直接使用高学习率
#     容易导致训练不稳定。warmup 给优化器时间适应梯度尺度。
# ============================================================

class WarmupCosineScheduler:
    """Warmup + Cosine Annealing，支持每个参数组独立的 lr_scale"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup = warmup_epochs
        self.total = total_epochs
        self.min_ratio = min_lr_ratio

    def step(self, epoch):
        """按 epoch（0-indexed）更新学习率"""
        if epoch < self.warmup:
            factor = (epoch + 1) / self.warmup
        else:
            progress = (epoch - self.warmup) / max(self.total - self.warmup, 1)
            factor = self.min_ratio + 0.5 * (1 - self.min_ratio) * (1 + np.cos(np.pi * progress))

        for pg in self.optimizer.param_groups:
            base_lr = pg.get('initial_lr', pg['lr'])
            scale = pg.get('lr_scale', 1.0)
            pg['lr'] = base_lr * factor * scale

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


# ============================================================
# 实验名称自动生成
#
#   根据参数组合生成唯一目录名，避免不同实验互相覆盖。
#   格式: tar(target)_r{rank}_a{alpha}_mask{0|1}_lr{lr}_opt(opt)_loss(loss)_foc{w}_bs{batch}x{accum}_{model}
#
#   示例: tar(cross_attn)_r4_a8_mask1_lr0.0001_opt(adamw)_loss(tversky)_foc0.5_bs1x4_sam2_hiera_t
# ============================================================

def build_exp_name(args) -> str:
    """根据命令行参数自动生成实验名称"""
    targets = '_'.join(args.lora_targets)
    mask = '1' if not args.no_mask_decoder else '0'
    model_name = Path(args.sam2_config).stem
    parts = [
        f"tar({targets})",
        f"r{args.lora_rank}",
        f"a{args.lora_alpha}",
        f"mask{mask}",
        f"lr{args.lr}",
        f"opt({args.optimizer})",
        f"loss({args.loss_type})",
        f"foc{args.focal_weight}",
        f"bs{args.batch_size}x{args.grad_accum}",
        f"{model_name}",
    ]
    return '_'.join(parts)


# ============================================================
# 命令行参数
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description='SAM2 LoRA 训练')

    # 数据
    p.add_argument('--data-root', default='/root/DataBscan')
    p.add_argument('--train-list', default='/root/e2e/train1.txt')
    p.add_argument('--test-list', default='/root/e2e/test1.txt')
    p.add_argument('--image-size', type=int, default=1024)
    p.add_argument('--clip-len', type=int, default=4)

    # SAM2 模型
    p.add_argument('--sam2-config', default='sam2_hiera_t.yaml')
    p.add_argument('--sam2-ckpt', default='checkpoints/sam2.1_hiera_tiny.pt')

    # LoRA 配置
    p.add_argument('--lora-targets', nargs='+',
                   default=['cross_attn'],
                   choices=['cross_attn', 'self_attn', 'memory_attn'],
                   help='LoRA 注入目标（可多选）')
    p.add_argument('--lora-rank', type=int, default=4)
    p.add_argument('--lora-alpha', type=int, default=8)
    p.add_argument('--no-mask-decoder', action='store_true',
                   help='冻结 mask decoder（只用 LoRA 微调）')

    # 训练超参数
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--warmup-epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=1,
                   help='clip 模式下固定为 1')
    p.add_argument('--grad-accum', type=int, default=4,
                   help='梯度累积步数（有效 batch_size = grad_accum）')
    p.add_argument('--grad-clip', type=float, default=1.0)

    # 损失函数
    p.add_argument('--loss-type', default='dice', choices=['dice', 'tversky'])
    p.add_argument('--focal-weight', type=float, default=0.5,
                   help='Focal loss 权重（0 = 只用 segmentation loss）')

    # 优化器
    p.add_argument('--optimizer', default='adamw',
                   choices=['adamw', 'muon_adam'])

    # 日志与输出
    p.add_argument('--output', default='auto',
                   help='输出目录（默认根据参数自动生成，避免覆盖）')
    p.add_argument('--val-every', type=int, default=5,
                   help='每 N 个 epoch 验证一次')
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--patience', type=int, default=10,
                   help='早停耐心值（0 = 不使用早停）')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--amp', action='store_true', default=True,
                   help='使用自动混合精度训练')
    p.add_argument('--no-amp', action='store_false', dest='amp')
    p.add_argument('--resume', default=None,
                   help='从 checkpoint 恢复训练')
    p.add_argument('--device', default='cuda')

    return p.parse_args()


# ============================================================
# 主训练函数
# ============================================================

def main():
    args = parse_args()

    # 固定随机种子（保证实验可复现）
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # 自动生成输出目录名（避免不同参数实验互相覆盖）
    if args.output == 'auto':
        args.output = f'results/phase3_lora/{build_exp_name(args)}'

    out = Path(args.output)
    if out.exists():
        print(f"\n  ⚠ 输出目录已存在: {out}")
        print(f"  内容可能被覆盖，Ctrl+C 取消，或继续...")
        import time; time.sleep(2)

    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    # 训练日志（同时输出到终端和文件，方便回溯历史训练情况）
    log_path = out / 'train.log'
    _log_fp = open(log_path, 'a', encoding='utf-8')
    import builtins as _bi
    _orig_print = print
    def _log_print(*a, **kw):
        _orig_print(*a, **kw)
        _orig_print(*a, **kw, file=_log_fp, flush=True)
    _bi.print = _log_print

    from datetime import datetime
    print(f"\n{'='*60}")
    print(f"训练开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # 读取数据列表（每行一个视频文件夹的相对路径）
    with open(args.train_list) as f:
        train_folders = [l.strip().replace('\\', '/') for l in f if l.strip()]
    with open(args.test_list) as f:
        test_folders = [l.strip().replace('\\', '/') for l in f if l.strip()]

    print("=" * 60)
    print("SAM2 LoRA 训练")
    print("=" * 60)
    print(f"  LoRA targets: {args.lora_targets}")
    print(f"  LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"  Mask decoder: {'训练' if not args.no_mask_decoder else '冻结'}")
    print(f"  Loss: {args.loss_type}, focal_weight: {args.focal_weight}")
    print(f"  LR: {args.lr}, Optimizer: {args.optimizer}")
    print(f"  Effective batch: {args.grad_accum}")
    print(f"  Train videos: {len(train_folders)}")
    print(f"  Test videos: {len(test_folders)}")

    # --- DataLoader ---

    # 训练 clip（50% 重叠，增加样本多样性）
    train_ds = SonarClipDataset(train_folders, args.data_root, 'uuv',
                                args.image_size, args.clip_len)

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    print(f"  Train clips: {len(train_ds)}, Test videos: {len(test_folders)}")

    # --- 模型 ---
    trainer = SAM2LoRATrainer(
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_ckpt,
        lora_config={
            'targets': args.lora_targets,
            'r': args.lora_rank,
            'alpha': args.lora_alpha,
        },
        train_mask_decoder=not args.no_mask_decoder,
        device=args.device,
    )

    # --- 优化器 ---
    use_dual = args.optimizer == 'muon_adam'
    if use_dual:
        # Muon+AdamW 双优化器策略：
        # - Muon 处理 2D/4D 权重矩阵（通过 Newton-Schulz 正交化，收敛更快）
        # - AdamW 处理 1D 偏置、LayerNorm、lora_A 等小矩阵
        from utils.muon import Muon
        groups = trainer.get_param_groups(lr_mult=10.0)
        optimizer = torch.optim.AdamW(
            groups[1]['params'], lr=args.lr, weight_decay=args.weight_decay
        )
        muon_opt = Muon(
            groups[0]['params'], lr=args.lr * 10, momentum=0.95
        ) if groups[0]['params'] else None
        second_opt = muon_opt
        # 记录初始学习率（供 scheduler 使用）
        for pg in optimizer.param_groups:
            pg['initial_lr'] = pg['lr']
            pg['lr_scale'] = 1.0
    else:
        optimizer = torch.optim.AdamW(
            trainer.get_trainable_params(), lr=args.lr, weight_decay=args.weight_decay
        )
        second_opt = None
        for pg in optimizer.param_groups:
            pg['initial_lr'] = pg['lr']
            pg['lr_scale'] = 1.0

    # --- 学习率调度 ---
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs)

    # --- 损失函数 & AMP ---
    loss_fn = CombinedLoss(
        seg_type=args.loss_type,
        seg_weight=1.0,
        focal_weight=args.focal_weight,
    )
    # 从 device 参数动态获取设备类型（不硬编码 'cuda'）
    scaler = torch.amp.GradScaler(args.device) if args.amp else None

    # --- 训练状态 ---
    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_niou': [],
               'val_dice': [], 'val_pd': [], 'lr': []}
    best_val_loss = float('inf')
    patience_cnt = 0
    start_epoch = 1

    # --- 恢复训练 ---
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        # 注意：checkpoint 保存的是 self.model.state_dict()（无 'model.' 前缀），
        # 所以加载到 trainer.model 而非 trainer
        trainer.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        history = ckpt.get('history', history)
        print(f"  恢复至 Epoch {start_epoch}")

    # ================================================================
    # 训练循环
    #
    # 梯度累积机制:
    #   1. 每个累积周期开始时: optimizer.zero_grad()
    #   2. 每个 batch: forward + backward（梯度在 .grad 中累加）
    #   3. 累积周期最后一步: grad_clip + optimizer.step()
    #
    # 关键：zero_grad() 必须在周期开始时调用，而非在 step() 之前。
    #       如果在 step() 之前调用，会清空前面 N-1 步累积的梯度。
    # ================================================================
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 60}")

        trainer.train()
        epoch_loss = 0.0
        epoch_frames = 0
        grad_steps = 0

        pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch}")
        for batch_idx, (images, points, gt) in enumerate(pbar):
            # DataLoader 加了 batch 维度 → squeeze 掉
            images = images.squeeze(0).to(args.device, non_blocking=True)
            points = points.squeeze(0)
            gt = gt.squeeze(0)

            # 梯度累积周期开始 — 清零梯度
            if batch_idx % args.grad_accum == 0:
                optimizer.zero_grad()
                if second_opt is not None:
                    second_opt.zero_grad()

            # 累积周期最后一步才更新参数
            skip_update = ((batch_idx + 1) % args.grad_accum != 0)

            result = trainer.train_step(
                clip_images=images,
                gt_centers=points,
                gt_masks=gt,
                optimizer=optimizer,
                loss_fn=loss_fn,
                grad_clip=args.grad_clip,
                scaler=scaler,
                second_optimizer=second_opt,
                skip_update=skip_update,
            )

            epoch_loss += result['loss']
            epoch_frames += len(result['frame_losses'])
            if not skip_update:
                grad_steps += 1

            pbar.set_postfix({
                'loss': f"{result['loss']:.4f}",
                'gsteps': grad_steps,
            })

        scheduler.step(epoch)

        avg_loss = epoch_loss / max(epoch_frames, 1)
        current_lr = scheduler.get_lr()
        history['train_loss'].append(avg_loss)
        history['lr'].append(current_lr)
        print(f"  平均 loss: {avg_loss:.4f}, LR: {current_lr:.2e}, grad steps: {grad_steps}")

        # --- 验证（官方 predictor + 仅 frame 0 GT 点 + clip 级传播） ---
        do_val = (epoch % args.val_every == 0) or (epoch == args.epochs)
        if do_val:
            vm = validate(trainer, test_folders, args.data_root,
                         args.image_size, args.clip_len,
                         loss_fn, args.device, args.sam2_config)
            history['val_loss'].append(vm['loss'])
            history['val_iou'].append(vm['iou'])
            history['val_niou'].append(vm['niou'])
            history['val_dice'].append(vm['dice'])
            history['val_pd'].append(vm['pd'])
            print(f"  验证: IoU={vm['iou']:.4f}, nIoU={vm['niou']:.4f}, "
                  f"Dice={vm['dice']:.4f}, Pd={vm['pd']:.3f}, Fa={vm['fa']:.5f}")

            # --- 保存最佳模型 ---
            if vm['loss'] < best_val_loss:
                best_val_loss = vm['loss']
                patience_cnt = 0

                # 训练 checkpoint（不合并 LoRA，用于继续训练）
                trainer.save_checkpoint(
                    str(ckpt_dir / 'best_train.pth'),
                    merge_lora=False,
                    extra={
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_metrics': vm,
                        'history': history,
                        'best_val_loss': best_val_loss,
                    },
                )
                # 推理 checkpoint（LoRA 已合并到权重，可直接被 SAM2 predictor 加载）
                trainer.save_checkpoint(
                    str(ckpt_dir / 'best_inference.pth'),
                    merge_lora=True,
                    extra={
                        'epoch': epoch,
                        'val_metrics': vm,
                    },
                )
                print(f"  ✓ 保存最佳模型 (val_loss={vm['loss']:.4f})")
            else:
                patience_cnt += 1
                if args.patience > 0 and patience_cnt >= args.patience:
                    print(f"  早停 (patience={args.patience})")
                    break

        # --- 定期保存 ---
        if epoch % args.save_every == 0:
            trainer.save_checkpoint(
                str(ckpt_dir / f'epoch_{epoch}.pth'),
                merge_lora=False,
                extra={
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,
                    'best_val_loss': best_val_loss,
                },
            )

    # --- 训练结束 ---
    json.dump(history, open(out / 'training_history.json', 'w'), indent=2)
    trainer.save_checkpoint(
        str(ckpt_dir / 'final.pth'),
        merge_lora=False,
        extra={
            'epoch': args.epochs,
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
        },
    )
    print(f"\n训练完成。最佳 val_loss: {best_val_loss:.4f}")
    print(f"输出目录: {out}")
    _log_fp.close()


if __name__ == '__main__':
    main()
