#!/usr/bin/env python
"""
SAM 2 LoRA 微调训练脚本 (单帧模式)

使用 build_sam2 底层模型进行微调，支持 LoRA + Mask Decoder。
将视频中的所有帧拆分为单张图片进行训练。

用法:
    python scripts/finetune_sam2_lora.py \
        --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
        --sam2-ckpt checkpoints/sam2.1_hiera_tiny.pt \
        --train-list /root/e2e/train1.txt \
        --output results/phase3_sam2_finetune \
        --epochs 50 \
        --lr 1e-5 \
        --lora-rank 4
"""

import os
import sys
import argparse
import json
import cv2
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '/root/autosam2')
sys.path.insert(0, '/root/e2e')

from models.sam2_finetuner_lora import SAM2LoRAFineTuner, CombinedLoss
from utils.metrics import NUDT_Metrics
from utils.heatmap import get_mask_center


def parse_args():
    parser = argparse.ArgumentParser(description='SAM 2 LoRA 微调训练')
    parser.add_argument('--sam2-config', type=str,
                        default='sam2_hiera_t.yaml')  # SAM2 Tiny（快速迭代）
    parser.add_argument('--sam2-ckpt', type=str,
                        default='checkpoints/sam2.1_hiera_tiny.pt')
    parser.add_argument('--data-root', type=str, default='/root/DataBscan')
    parser.add_argument('--train-list', type=str, default='/root/e2e/train1.txt')
    parser.add_argument('--test-list', type=str, default='/root/e2e/test1.txt')
    parser.add_argument('--output', type=str, default='results/phase3_sam2_lora')
    parser.add_argument('--target-label', type=str, default='uuv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lora-rank', type=int, default=4)
    parser.add_argument('--lora-alpha', type=int, default=8)
    parser.add_argument('--optimizer', type=str, default='muon_adam',
                        choices=['adamw', 'muon_adam'],
                        help='优化器: adamw 或 muon_adam (Muon+AdamW混合)')
    parser.add_argument('--loss-type', type=str, default='tversky',
                        choices=['dice', 'tversky'],
                        help='分割损失函数: dice 或 tversky (对小目标更友好)')
    parser.add_argument('--image-size', type=int, default=1024,
                        help='训练图像分辨率（必须为1024，SAM2的内部分辨率硬编码）')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='批大小（Tiny建议8; Large建议3）')
    parser.add_argument('--grad-accum', type=int, default=8,
                        help='梯度累积步数（模拟batch_size*8，提升GPU利用率）')
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--val-every', type=int, default=5,
                        help='每N个epoch验证一次（单帧loss，非端到端指标）')
    parser.add_argument('--patience', type=int, default=15,
                        help='早停耐心值：连续N轮验证loss不降则停止（0=禁用）')
    parser.add_argument('--warmup-epochs', type=int, default=3,
                        help='学习率预热轮数（逐步从0升到目标lr）')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='启用混合精度训练（省显存+加速）')
    parser.add_argument('--no-amp', action='store_false', dest='amp',
                        help='禁用混合精度')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（保证实验可复现）')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='从checkpoint恢复训练（best_train.pth路径）')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def apply_augmentation(batch_frames):
    """
    声呐时序数据增强（在 batch 级别随机应用）

    增强策略：
    - 水平翻转 (50%): 声呐沿轨迹方向对称
    - 时间逆序 (30%): 目标可以正向或反向移动
    - 亮度抖动 (50%): 模拟不同 SNR 条件
    - 随机平移 (30%): 模拟目标位置变化

    注意：不做垂直翻转，因为声呐的深度/距离轴不对称
    """
    if np.random.random() < 0.5:
        # 水平翻转（bearing 轴对称）
        for f in batch_frames:
            f['image'] = torch.flip(f['image'], dims=[-1])  # 翻转 W 轴
            f['gt_mask'] = torch.flip(f['gt_mask'], dims=[-1])
            # 修正 prompt 点坐标
            W = f['image'].shape[-1]
            f['point'][..., 0] = W - f['point'][..., 0]

    if np.random.random() < 0.3:
        # 时间逆序（帧序列反转）
        batch_frames.reverse()

    if np.random.random() < 0.5:
        # 亮度/对比度抖动（模拟不同 SNR）
        brightness = 0.85 + np.random.random() * 0.3   # [0.85, 1.15]
        contrast = 0.9 + np.random.random() * 0.2      # [0.9, 1.1]
        for f in batch_frames:
            f['image'] = torch.clamp((f['image'] * brightness + (1 - brightness) * 0.5) * contrast, -3, 3)

    if np.random.random() < 0.3 and len(batch_frames) > 0:
        # 小幅度平移（±5% 图像尺寸）
        W_img = batch_frames[0]['image'].shape[-1]
        H_img = batch_frames[0]['image'].shape[-2]
        shift_x = int(0.05 * np.random.randn() * W_img)
        shift_y = int(0.05 * np.random.randn() * H_img)
        for f in batch_frames:
            f['image'] = torch.roll(f['image'], shifts=(shift_y, shift_x), dims=(-2, -1))
            f['gt_mask'] = torch.roll(f['gt_mask'], shifts=(shift_y, shift_x), dims=(-2, -1))
            f['point'][0] += shift_x
            f['point'][1] += shift_y

    return batch_frames


def load_video_frames(video_folder, target_label, image_size):
    """
    加载单个视频的所有有效帧（按需加载，避免OOM）

    Args:
        video_folder: 视频文件夹路径
        target_label: 目标标签
        image_size: 训练图像分辨率

    Returns:
        List[Dict]: 帧数据列表
    """
    img_files = sorted([f for f in os.listdir(video_folder)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    frames = []

    for img_file in img_files:
        img_path = os.path.join(video_folder, img_file)
        json_name = os.path.splitext(img_file)[0] + '.json'
        json_path = os.path.join(video_folder, json_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for s in data.get('shapes', []):
                    if s['label'] == target_label:
                        points = np.array(s['points'], dtype=np.float32)
                        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
            except Exception:
                continue

        if mask.sum() == 0:
            continue

        center = get_mask_center(torch.from_numpy(mask))
        if center is None:
            continue

        # Resize
        img_resized = cv2.resize(img, (image_size, image_size))
        mask_resized = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

        scale_x = image_size / w
        scale_y = image_size / h

        # 转换为tensor: (3, H, W), [0, 1], ImageNet归一化
        img_tensor = torch.from_numpy(img_resized).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).expand(3, -1, -1)
        img_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - img_mean) / img_std

        frames.append({
            'image': img_tensor,
            'point': np.array([center[0] * scale_x, center[1] * scale_y], dtype=np.float32),
            'point_label': np.array([1], dtype=np.int64),
            'gt_mask': torch.from_numpy(mask_resized).float().unsqueeze(0)
        })

    return frames


def compute_iou(pred_prob, gt_mask, threshold=0.5):
    """单帧 IoU（pred_prob 是 sigmoid 输出，需 threshold=0.5 二值化）"""
    pred_bin = (pred_prob > threshold).float()
    intersection = (pred_bin * gt_mask).sum()
    union = pred_bin.sum() + gt_mask.sum() - intersection
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (intersection / union).item()

def compute_dice(pred_prob, gt_mask, threshold=0.5):
    """单帧 Dice 系数"""
    pred_bin = (pred_prob > threshold).float()
    intersection = (pred_bin * gt_mask).sum()
    total = pred_bin.sum() + gt_mask.sum()
    if total == 0:
        return 1.0
    return (2 * intersection / total).item()

def validate(model, test_folders, data_root, target_label, image_size, loss_fn, device):
    """验证：计算单帧 loss、IoU、Dice（不跑端到端视频跟踪）"""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_frames = 0

    with torch.no_grad():
        for folder in test_folders:
            folder_path = os.path.join(data_root, folder)
            if not os.path.isdir(folder_path):
                continue

            video_frames = load_video_frames(folder_path, target_label, image_size)
            if not video_frames:
                continue

            for frame_data in video_frames:
                image = frame_data['image'].unsqueeze(0).to(device)
                point = torch.from_numpy(frame_data['point']).unsqueeze(0).unsqueeze(0).to(device)  # (2,)→(1,1,2)
                point_label = torch.from_numpy(frame_data['point_label']).unsqueeze(0).to(device)  # (1,)→(1,1)
                gt_mask = frame_data['gt_mask'].unsqueeze(0).to(device)

                low_res_masks, _ = model.forward_single_frame(image, point, point_label)
                pred_masks = torch.nn.functional.interpolate(
                    low_res_masks, size=gt_mask.shape[-2:], mode='bilinear', align_corners=False
                )
                pred_sigmoid = torch.sigmoid(pred_masks)

                loss = loss_fn(pred_masks, gt_mask)
                total_loss += loss.item()

                # 计算单帧分割指标
                total_iou += compute_iou(pred_sigmoid, gt_mask)
                total_dice += compute_dice(pred_sigmoid, gt_mask)
                total_frames += 1

            del video_frames

    n = max(total_frames, 1)
    return {
        'loss': total_loss / n,
        'iou': total_iou / n,
        'dice': total_dice / n,
        'num_frames': total_frames
    }


def main():
    args = parse_args()

    # 设置随机种子（保证实验可复现）
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # cuDNN 自动选择最优算法（固定输入尺寸时可加速）
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # 用 benchmark 时需关闭 deterministic

    # 混合精度训练（节省显存 + 加速）
    scaler = torch.amp.GradScaler('cuda') if args.device == 'cuda' else None

    print("=" * 60)
    print("SAM 2 LoRA 微调训练")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    # 加载列表
    with open(args.train_list, 'r') as f:
        train_folders = [line.strip().replace('\\', '/') for line in f if line.strip()]

    with open(args.test_list, 'r') as f:
        test_folders = [line.strip().replace('\\', '/') for line in f if line.strip()]

    print(f"训练视频: {len(train_folders)}")
    print(f"验证视频: {len(test_folders)}")
    print(f"学习率: {args.lr}")
    print(f"LoRA秩: {args.lora_rank}")
    print(f"图像大小: {args.image_size}")

    # 构建模型
    print(f"\n加载模型: {args.sam2_ckpt}")
    model = SAM2LoRAFineTuner(
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_ckpt,
        device=args.device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )

    # 优化器
    if args.optimizer == 'muon_adam':
        from utils.muon import Muon
        MUON_LR_MULT = 10  # Muon 学习率倍增因子（借鉴LVNet）

        muon_params = []
        adam_params = []
        for name, param in model.model.named_parameters():
            if not param.requires_grad:
                continue
            # 2D/4D 参数用 Muon（卷积权重、线性层权重）
            if param.ndim in [2, 4] and 'bias' not in name and 'dwconv' not in name:
                muon_params.append(param)
            else:
                adam_params.append(param)

        muon_lr = args.lr * MUON_LR_MULT
        print(f"优化器: Muon+AdamW (Muon LR={muon_lr:.2e}, AdamW LR={args.lr:.2e})")
        print(f"  Muon 参数: {len(muon_params)} 组, AdamW 参数: {len(adam_params)} 组")

        optimizer = Muon(muon_params, lr=muon_lr, momentum=0.95) if muon_params else None
        adam_opt = torch.optim.AdamW(adam_params, lr=args.lr, weight_decay=args.weight_decay) if adam_params else None
    else:
        print(f"优化器: AdamW (LR={args.lr:.2e})")
        optimizer = torch.optim.AdamW(
            model.get_trainable_parameters(),
            lr=args.lr, weight_decay=args.weight_decay
        )
        adam_opt = None

    # 学习率调度器（warmup + cosine）
    def warmup_cosine_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress)) * (1 - 0.01) + 0.01

    # 双优化器时各自独立调度
    use_dual_opt = (args.optimizer == 'muon_adam')
    if use_dual_opt:
        scheduler_muon = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_lambda)
        scheduler_adam = torch.optim.lr_scheduler.LambdaLR(adam_opt, warmup_cosine_lambda) if adam_opt else None
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_lambda)

    # 损失函数
    loss_fn = CombinedLoss(
        loss_type=args.loss_type,
        tversky_alpha=0.7,
        tversky_beta=0.3
    )
    print(f"损失函数: {args.loss_type} + Focal")

    # 训练历史
    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_dice': [], 'learning_rate': []}

    best_val_loss = float('inf')
    start_epoch = 1
    patience_counter = 0  # 早停计数器

    # 梯度裁剪阈值
    GRAD_CLIP = 1.0

    # 恢复训练
    if args.resume_from:
        print(f"\n从checkpoint恢复: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=args.device)
        model.model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        val_metrics_saved = ckpt.get('val_metrics', {})
        best_val_loss = val_metrics_saved.get('loss', ckpt.get('val_loss', float('inf')))
        if 'train_loss' in ckpt:
            history['train_loss'].append(ckpt['train_loss'])
        print(f"恢复至 Epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # 训练循环
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 60}")

        model.train()
        epoch_loss = 0.0
        total_frames = 0

        # 逐视频加载训练帧（避免OOM）
        np.random.shuffle(train_folders)

        pbar = tqdm(train_folders, desc=f"训练 Epoch {epoch}")
        for folder in pbar:
            folder_path = os.path.join(args.data_root, folder)
            if not os.path.isdir(folder_path):
                continue

            try:
                video_frames = load_video_frames(folder_path, args.target_label, args.image_size)
                if not video_frames:
                    continue

                # 视频内打乱
                np.random.shuffle(video_frames)

                # 批次处理：batch_size 帧拼成一个 tensor
                batch_frames = []
                accum_count = 0

                for i, frame_data in enumerate(video_frames):
                    batch_frames.append(frame_data)

                    if len(batch_frames) < args.batch_size and i < len(video_frames) - 1:
                        continue

                    # 数据增强（仅训练时）
                    batch_frames = apply_augmentation(batch_frames)

                    # 拼接 batch
                    images = torch.stack([f['image'] for f in batch_frames]).to(args.device)
                    points = torch.from_numpy(np.stack([f['point'] for f in batch_frames])).unsqueeze(1).to(args.device)  # (B,2)→(B,1,2)
                    point_labels = torch.from_numpy(np.stack([f['point_label'] for f in batch_frames])).to(args.device)  # (B,1) - 2 dims
                    gt_masks = torch.stack([f['gt_mask'] for f in batch_frames]).to(args.device)

                    # 双优化器时，在每个累积周期开始时清零梯度
                    if use_dual_opt and accum_count == 0:
                        optimizer.zero_grad()
                        if adam_opt:
                            adam_opt.zero_grad()

                    result = model.train_step(images, points, point_labels, gt_masks,
                                              optimizer, loss_fn, grad_clip=GRAD_CLIP,
                                              scaler=scaler if args.amp else None,
                                              second_optimizer=adam_opt if use_dual_opt else None,
                                              skip_step=(accum_count + 1 < args.grad_accum))
                    accum_count += 1
                    if accum_count >= args.grad_accum:
                        accum_count = 0

                    batch_frames = []  # 清空准备下一批

                    if result['valid']:
                        epoch_loss += result['loss']
                        total_frames += 1
                        pbar.set_postfix({'loss': f"{result['loss']:.4f}"})

                # 释放视频帧内存
                del video_frames

            except Exception as e:
                print(f"\n跳过 {folder}: {e}")
                continue

        # 更新学习率
        if use_dual_opt:
            scheduler_muon.step()
            if scheduler_adam:
                scheduler_adam.step()
            current_lr = scheduler_muon.get_last_lr()[0]
        else:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

        avg_loss = epoch_loss / max(total_frames, 1)
        history['train_loss'].append(avg_loss)
        history['learning_rate'].append(current_lr)

        print(f"平均训练损失: {avg_loss:.4f}")
        print(f"学习率: {current_lr:.2e}")

        # 定期验证（单帧指标，非端到端视频跟踪）
        do_validate = (epoch % args.val_every == 0) or (epoch == args.epochs)
        val_metrics = {'loss': float('inf'), 'iou': 0.0, 'dice': 0.0, 'num_frames': 0}

        if do_validate:
            val_metrics = validate(model, test_folders, args.data_root,
                                  args.target_label, args.image_size, loss_fn, args.device)
            history['val_loss'].append(val_metrics['loss'])
            history['val_iou'].append(val_metrics['iou'])
            history['val_dice'].append(val_metrics['dice'])
            print(f"验证: loss={val_metrics['loss']:.4f}, IoU={val_metrics['iou']:.4f}, Dice={val_metrics['dice']:.4f}")

        # 基于验证 loss 保存最佳模型 + 早停检查
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            best_train_path = checkpoint_dir / 'best_train.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'val_metrics': val_metrics,
                'lora_rank': args.lora_rank
            }, best_train_path)

            best_infer_path = checkpoint_dir / 'best_inference.pth'
            model.save_for_inference(str(best_infer_path), merge_lora=True)

            print(f"✓ 保存最佳模型: train_loss={avg_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
                  f"IoU={val_metrics['iou']:.4f}")
        elif do_validate and args.patience > 0:
            patience_counter += 1
            print(f"  验证 loss 未改善 ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"\n早停！验证 loss 连续 {args.patience} 轮未下降")
                break

        # 定期保存
        if epoch % args.save_every == 0:
            ckpt_path = checkpoint_dir / f'epoch_{epoch}_train.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print(f"✓ 保存checkpoint: {ckpt_path}")

        # 第一个epoch后打印梯度摘要
        if epoch == 1:
            lora_grads = []
            decoder_grads = []
            for name, param in model.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    gn = param.grad.norm().item()
                    if 'lora' in name.lower():
                        lora_grads.append(gn)
                    elif 'mask_decoder' in name.lower():
                        decoder_grads.append(gn)
            if lora_grads:
                print(f"LoRA 梯度范围: {min(lora_grads):.6f} ~ {max(lora_grads):.6f}")
            if decoder_grads:
                print(f"Decoder 梯度范围: {min(decoder_grads):.6f} ~ {max(decoder_grads):.6f}")

    # 保存最终模型
    final_path = checkpoint_dir / 'final.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)

    # 保存训练历史
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"最佳验证loss: {best_val_loss:.4f}")
    print(f"最终模型: {final_path}")
    print(f"训练历史: {history_path}")


if __name__ == '__main__':
    main()
