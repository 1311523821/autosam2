#!/usr/bin/env python
"""
引导头训练脚本 (Guidance Head Training Script)

训练轻量级时空引导头，用于自动生成SAM 2的Prompt点。

用法:
    python train_guidance.py --config configs/default.yaml
    python train_guidance.py --epochs 50 --batch-size 4
"""

import os
import sys
import argparse
import yaml
import time
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, '/root/autosam2')
sys.path.insert(0, '/root/e2e')

from models.guidance_head import GuidanceHead, build_guidance_head
from models.guidance_head_swin import GuidanceHeadSwin, build_guidance_head_swin
from data.dataset import VideoHeatmapDataset, heatmap_collate_fn, get_heatmap_dataloader
from utils.losses import HeatmapFocalLoss, build_loss
from utils.heatmap import extract_point_from_heatmap, get_mask_center
from utils.muon import Muon


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练引导头模型')

    # 数据参数
    parser.add_argument('--data-root', type=str, default='/root/DataBscan',
                        help='数据根目录')
    parser.add_argument('--train-list', type=str, default='/root/e2e/train1.txt',
                        help='训练集列表文件')
    parser.add_argument('--val-list', type=str, default=None,
                        help='验证集列表文件（非K-Fold模式下使用）')
    parser.add_argument('--test-list', type=str, default='/root/e2e/test1.txt',
                        help='测试集列表文件（训练后独立评估）')
    parser.add_argument('--target-label', type=str, default='uuv',
                        help='目标标签')

    # 模型参数 (基于消融实验最佳配置 exp6)
    parser.add_argument('--embed-dim', type=int, default=64,
                        help='嵌入维度')
    parser.add_argument('--hidden-dim', type=int, default=32,
                        help='隐藏层维度')
    parser.add_argument('--clip-len', type=int, default=4,
                        help='时序帧数')

    # Swin 架构参数
    parser.add_argument('--use-swin', action='store_true',
                        help='使用 STSF + Swin 3D 混合架构')
    parser.add_argument('--swin-depth', type=int, default=1,
                        help='Swin Block 深度')
    parser.add_argument('--swin-heads', type=int, default=2,
                        help='Swin 注意力头数')
    parser.add_argument('--use-spatial-attn', action='store_true',
                        help='在 Swin Block 中使用空间注意力')
    parser.add_argument('--no-conv3d-mlp', action='store_true',
                        help='禁用 Conv3D MLP（节省显存）')

    # 训练参数 (基于消融实验最佳配置 exp6)
    parser.add_argument('--batch-size', type=int, default=8,
                        help='批大小')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='数据加载工作进程数')
    parser.add_argument('--accum-steps', type=int, default=2,
                        help='梯度累积步数')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'muon_adam'],
                        help='优化器类型: adamw 或 muon_adam (Muon+AdamW混合)')

    # 损失函数参数
    parser.add_argument('--loss-type', type=str, default='heatmap_focal',
                        choices=['heatmap_focal', 'mse'],
                        help='损失函数类型')
    parser.add_argument('--focal-alpha', type=int, default=2,
                        help='Focal损失alpha参数')
    parser.add_argument('--focal-beta', type=int, default=4,
                        help='Focal损失beta参数')
    parser.add_argument('--heatmap-sigma', type=float, default=10.0,
                        help='高斯热力图标准差')
    parser.add_argument('--img-size', type=int, nargs=2, default=[256, 256],
                        metavar=('H', 'W'),
                        help='目标图像尺寸 (H W)，例如 --img-size 256 256')
    parser.add_argument('--threshold-radius', type=int, default=5,
                        help='命中阈值（像素）')

    # 调度器参数
    parser.add_argument('--warmup-epochs', type=int, default=3,
                        help='预热轮数')
    parser.add_argument('--patience', type=int, default=15,
                        help='早停耐心值')

    # 输出参数
    parser.add_argument('--output', type=str, default='checkpoints/guidance_head',
                        help='输出目录')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='日志打印间隔')

    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径（覆盖命令行参数）')

    # K折交叉验证参数
    parser.add_argument('--kfold', type=int, default=5,
                        help='K折交叉验证折数（0表示不使用K折）')
    parser.add_argument('--fold', type=int, default=-1,
                        help='指定训练某一折（-1表示训练所有折）')

    # 随机种子
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子，用于可复现性')

    return parser.parse_args()


def set_seed(seed: int):
    """
    设置全局随机种子，确保可复现性

    参数:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU

    # 设置 cuDNN 的确定性模式（会降低性能，但确保可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"随机种子已设置: {seed}")


def load_config(args):
    """加载配置文件"""
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 用配置文件覆盖默认参数
        if 'data' in config:
            args.data_root = config['data'].get('root_dir', args.data_root)
            args.train_list = config['data'].get('train_list', args.train_list)
            args.val_list = config['data'].get('test_list', args.val_list)
            args.target_label = config['data'].get('target_label', args.target_label)

        if 'model' in config and 'guidance_head' in config['model']:
            gh = config['model']['guidance_head']
            args.embed_dim = gh.get('embed_dim', args.embed_dim)
            args.clip_len = gh.get('clip_len', args.clip_len)

        if 'training' in config:
            t = config['training']
            args.batch_size = t.get('batch_size', args.batch_size)
            args.epochs = t.get('epochs', args.epochs)
            args.lr = t.get('lr', args.lr)
            args.weight_decay = t.get('weight_decay', args.weight_decay)
            args.warmup_epochs = t.get('warmup_epochs', args.warmup_epochs)
            args.patience = t.get('patience', args.patience)

            if 'loss' in t:
                args.focal_alpha = t['loss'].get('focal_gamma', args.focal_alpha)
            if 'heatmap' in t:
                args.heatmap_sigma = t['heatmap'].get('sigma', args.heatmap_sigma)

    return args


def read_video_list(txt_path):
    """读取视频列表文件"""
    with open(txt_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def split_kfold(video_list, k=5, seed=42):
    """
    按视频ID分组划分K折，避免数据泄露

    确保同一视频的所有clip都在同一折中，避免数据泄露。

    参数:
        video_list: 视频路径列表，格式如 "xxx/DataRecord_xxx"
        k: 折数
        seed: 随机种子

    返回:
        folds: 列表，每个元素是 (train_indices, val_indices)
    """
    np.random.seed(seed)

    # 提取视频ID（从路径中提取视频名称）
    video_ids = {}
    for idx, path in enumerate(video_list):
        # 路径格式: "day1/testing/jin/DataRecord_2025-12-08_132412"
        # 视频名称是最后一级目录
        parts = path.replace('\\', '/').rstrip('/').split('/')
        video_name = parts[-1] if parts else path

        if video_name not in video_ids:
            video_ids[video_name] = []
        video_ids[video_name].append(idx)

    # 获取所有唯一的视频ID
    unique_videos = list(video_ids.keys())
    np.random.shuffle(unique_videos)

    print(f"发现 {len(unique_videos)} 个唯一视频，共 {len(video_list)} 个clip")

    # 检查是否有足够的视频进行K折划分
    if len(unique_videos) < k:
        print(f"⚠️ 视频数量 ({len(unique_videos)}) 小于 K ({k})，将使用 Leave-One-Out 策略")
        k = len(unique_videos)

    # 按视频ID划分K折
    folds = []
    fold_size = len(unique_videos) // k

    for i in range(k):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < k - 1 else len(unique_videos)
        val_videos = unique_videos[val_start:val_end]
        train_videos = unique_videos[:val_start] + unique_videos[val_end:]

        # 收集所有clip索引
        train_indices = []
        for vid in train_videos:
            train_indices.extend(video_ids[vid])

        val_indices = []
        for vid in val_videos:
            val_indices.extend(video_ids[vid])

        folds.append((np.array(train_indices), np.array(val_indices)))

    return folds


def create_dataloader_from_indices(video_list, indices, root_dir, clip_len,
                                    target_label, is_train, batch_size,
                                    num_workers, heatmap_sigma, img_size=None, seed=None):
    """
    从视频列表和索引创建数据加载器

    参数:
        video_list: 完整视频路径列表
        indices: 要使用的视频索引
        其他参数同 get_heatmap_dataloader

    返回:
        DataLoader
    """
    # 创建临时文件
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for idx in indices:
            f.write(video_list[idx] + '\n')
        temp_path = f.name

    # 创建数据加载器
    loader = get_heatmap_dataloader(
        txt_path=temp_path,
        root_dir=root_dir,
        clip_len=clip_len,
        target_label=target_label,
        is_train=is_train,
        batch_size=batch_size,
        num_workers=num_workers,
        heatmap_sigma=heatmap_sigma,
        img_size=img_size,
        seed=seed
    )

    # 删除临时文件（DataLoader已经读取完成）
    os.unlink(temp_path)

    return loader


def evaluate_localization_accuracy(model, dataloader, device, threshold_radius=10):
    """
    评估点定位准确率

    参数:
        model: 引导头模型
        dataloader: 数据加载器
        device: 设备
        threshold_radius: 容忍半径（像素）

    返回:
        hit_rate: 命中率
        mean_distance: 平均距离
        mean_confidence: 平均置信度
    """
    model.eval()
    total_samples = 0
    total_hits = 0
    total_distance = 0.0
    total_confidence = 0.0

    with torch.no_grad():
        for frames, heatmap_gt, masks, centers in dataloader:
            frames = frames.to(device)
            centers = centers.to(device)

            # 前向传播
            pred_logits = model(frames)

            # 提取预测点
            pred_heatmap = torch.sigmoid(pred_logits)
            B, C, H, W = pred_heatmap.shape
            pred_flat = pred_heatmap.view(B, -1)
            max_vals, max_idx = torch.max(pred_flat, dim=1)

            pred_y = max_idx // W
            pred_x = max_idx % W

            # 计算距离
            gt_x, gt_y = centers[:, 0], centers[:, 1]
            distance = torch.sqrt((pred_x.float() - gt_x)**2 + (pred_y.float() - gt_y)**2)

            # 统计
            hits = (distance <= threshold_radius).sum().item()
            total_hits += hits
            total_distance += distance.sum().item()
            total_confidence += max_vals.sum().item()
            total_samples += B

    hit_rate = total_hits / total_samples if total_samples > 0 else 0
    mean_distance = total_distance / total_samples if total_samples > 0 else 0
    mean_confidence = total_confidence / total_samples if total_samples > 0 else 0

    return hit_rate, mean_distance, mean_confidence


def evaluate_on_testset(model_path: str, test_list: str, args, device) -> dict:
    """
    在测试集上评估模型

    参数:
        model_path: 模型权重路径
        test_list: 测试集列表文件
        args: 训练参数
        device: 设备

    返回:
        包含评估结果的字典
    """
    from models.guidance_head import GuidanceHead
    from models.guidance_head_swin import GuidanceHeadSwin

    # 1. 创建模型
    if args.use_swin:
        model = GuidanceHeadSwin(
            in_chans=1,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_frame=args.clip_len,
            use_conv3d_mlp=not args.no_conv3d_mlp
        )
    else:
        model = GuidanceHead(
            in_chans=1,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_frame=args.clip_len
        )

    # 2. 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 3. 创建测试数据集
    test_loader = get_heatmap_dataloader(
        txt_path=test_list,
        root_dir=args.data_root,
        clip_len=args.clip_len,
        target_label=args.target_label,
        is_train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        heatmap_sigma=args.heatmap_sigma,
        img_size=tuple(args.img_size) if args.img_size else None,
        seed=args.seed
    )

    # 4. 计算评估阈值
    threshold = args.threshold_radius if args.threshold_radius else (
        5 if args.img_size else 10
    )

    # 5. 评估
    hit_rate, mean_dist, confidence = evaluate_localization_accuracy(
        model, test_loader, device, threshold_radius=threshold
    )

    return {
        'hit_rate': hit_rate,
        'mean_distance': mean_dist,
        'confidence': confidence,
        'threshold': threshold,
        'num_samples': len(test_loader.dataset)
    }


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scaler,
    device,
    accum_steps,
    epoch,
    log_interval
):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

    for step, (frames, heatmap_gt, masks, centers) in enumerate(progress_bar):
        frames = frames.to(device)
        heatmap_gt = heatmap_gt.to(device)

        # 混合精度训练
        with torch.amp.autocast('cuda'):
            pred_logits = model(frames)
            loss = criterion(pred_logits, heatmap_gt) / accum_steps

        scaler.scale(loss).backward()

        # 梯度累积
        if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
            # 支持多优化器 (Muon-Adam)
            if isinstance(optimizer, list):
                for opt in optimizer:
                    scaler.step(opt)
                    opt.zero_grad()
            else:
                scaler.step(optimizer)
                optimizer.zero_grad()
            scaler.update()

        # 记录损失
        real_loss = loss.item() * accum_steps
        total_loss += real_loss
        num_batches += 1

        # 更新进度条
        if (step + 1) % log_interval == 0:
            progress_bar.set_postfix({'loss': f'{real_loss:.4f}'})

    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for frames, heatmap_gt, masks, centers in dataloader:
            frames = frames.to(device)
            heatmap_gt = heatmap_gt.to(device)

            pred_logits = model(frames)
            loss = criterion(pred_logits, heatmap_gt)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def get_exp_name_prefix(args):
    """
    生成统一的实验名称前缀（模仿LVNet_ablation.py风格）

    格式: enc({encoder})_embed({embed_dim})_hidden({hidden_dim})_opt({optimizer})_lr({lr})_loss({loss})_sigma({sigma})_kfold({k})
    示例: enc(stsf_swin)_embed(32)_hidden(16)_opt(muon_adam)_lr(5e-4)_loss(focal_a2)_sigma(15)_kfold(5)
    """
    # 编码器类型
    encoder = "stsf_swin" if args.use_swin else "stsf"

    # 损失函数
    if args.loss_type == 'heatmap_focal':
        loss = f"focal_a{args.focal_alpha}"
    else:
        loss = args.loss_type

    # 学习率简化表示
    lr_str = f"{args.lr:.0e}".replace('-0', '-')

    # K折
    kfold_str = f"_kfold({args.kfold})" if args.kfold > 0 else ""

    return f"enc({encoder})_embed({args.embed_dim})_hidden({args.hidden_dim})_opt({args.optimizer})_lr({lr_str})_loss({loss})_sigma({args.heatmap_sigma:.0f}){kfold_str}"


def main():
    """主函数"""
    args = parse_args()
    args = load_config(args)

    # 设置随机种子（可复现性）
    set_seed(args.seed)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建输出目录（包含实验名称）
    exp_name = get_exp_name_prefix(args)
    output_dir = Path(args.output) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f"配置已保存到 {config_path}")

    # K折交叉验证模式
    if args.kfold > 0:
        train_with_kfold(args, device, output_dir)
    else:
        # 原始单次训练模式
        train_single(args, device, output_dir)


def train_single(args, device, output_dir):
    """单次训练（非K折）"""
    # 创建日志目录
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # 创建数据集
    print("\n" + "=" * 50)
    print("加载数据集...")
    print("=" * 50)

    train_loader = get_heatmap_dataloader(
        txt_path=args.train_list,
        root_dir=args.data_root,
        clip_len=args.clip_len,
        target_label=args.target_label,
        is_train=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        heatmap_sigma=args.heatmap_sigma,
        img_size=tuple(args.img_size) if args.img_size else None,
        seed=args.seed
    )

    val_loader = get_heatmap_dataloader(
        txt_path=args.val_list,
        root_dir=args.data_root,
        clip_len=args.clip_len,
        target_label=args.target_label,
        is_train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        heatmap_sigma=args.heatmap_sigma,
        img_size=tuple(args.img_size) if args.img_size else None,
        seed=args.seed
    )

    # 创建模型、优化器等
    model, criterion, optimizer, scheduler, scaler = create_model_and_optimizer(args, device)

    # 训练循环
    best_hit_rate = run_training(
        model, train_loader, val_loader, criterion, optimizer, scheduler, scaler,
        device, args, output_dir, writer
    )

    writer.close()
    print("\n" + "=" * 50)
    print("训练完成！")
    print(f"最佳 Hit Rate@10: {best_hit_rate:.2%}")
    print(f"模型保存到: {output_dir}")
    print("=" * 50)


def train_with_kfold(args, device, output_dir):
    """K折交叉验证训练"""
    print("\n" + "=" * 50)
    print(f"开始 {args.kfold} 折交叉验证训练")
    print("=" * 50)

    # 读取完整视频列表
    video_list = read_video_list(args.train_list)
    print(f"训练集视频数量: {len(video_list)}")

    # 划分K折
    folds = split_kfold(video_list, k=args.kfold, seed=args.seed)

    # 存储每折结果
    fold_results = []

    # 确定要训练的折
    if args.fold >= 0 and args.fold < args.kfold:
        folds_to_train = [args.fold]
    else:
        folds_to_train = range(args.kfold)

    for fold_idx in folds_to_train:
        print("\n" + "=" * 60)
        print(f"Fold {fold_idx + 1}/{args.kfold}")
        print("=" * 60)

        # 获取当前折的索引
        train_indices, val_indices = folds[fold_idx]
        print(f"训练视频: {len(train_indices)} 个")
        print(f"验证视频: {len(val_indices)} 个")

        # 创建当前折的输出目录
        fold_dir = output_dir / f'fold_{fold_idx + 1}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        # 创建日志目录
        log_dir = fold_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))

        # 创建数据加载器
        train_loader = create_dataloader_from_indices(
            video_list, train_indices, args.data_root, args.clip_len,
            args.target_label, True, args.batch_size,
            args.num_workers, args.heatmap_sigma,
            img_size=tuple(args.img_size) if args.img_size else None,
            seed=args.seed
        )

        val_loader = create_dataloader_from_indices(
            video_list, val_indices, args.data_root, args.clip_len,
            args.target_label, False, args.batch_size,
            args.num_workers, args.heatmap_sigma,
            img_size=tuple(args.img_size) if args.img_size else None,
            seed=args.seed
        )

        # 创建模型、优化器等
        model, criterion, optimizer, scheduler, scaler = create_model_and_optimizer(args, device)

        # 训练
        best_hit_rate = run_training(
            model, train_loader, val_loader, criterion, optimizer, scheduler, scaler,
            device, args, fold_dir, writer, fold_idx=fold_idx
        )

        writer.close()
        fold_results.append(best_hit_rate)
        print(f"\nFold {fold_idx + 1} 最佳 Hit Rate@10: {best_hit_rate:.2%}")

        # 在测试集上评估当前 Fold 的最佳模型
        if args.test_list and os.path.exists(args.test_list):
            best_model_path = fold_dir / "best.pth"
            if best_model_path.exists():
                print(f"\n[测试集评估] Fold {fold_idx + 1}")
                test_results = evaluate_on_testset(
                    str(best_model_path), args.test_list, args, device
                )
                print(f"  测试集 Hit Rate@{test_results['threshold']}: {test_results['hit_rate']:.2%}")
                print(f"  平均距离: {test_results['mean_distance']:.2f}px")

                # 保存该 Fold 的测试集结果
                test_result_file = fold_dir / "test_results.txt"
                with open(test_result_file, 'w', encoding='utf-8') as f:
                    f.write(f"Fold {fold_idx + 1} 测试集评估结果\n")
                    f.write("=" * 40 + "\n")
                    f.write(f"测试集: {args.test_list}\n")
                    f.write(f"样本数: {test_results['num_samples']}\n")
                    f.write(f"评估阈值: {test_results['threshold']}px\n")
                    f.write(f"Hit Rate: {test_results['hit_rate']:.4f}\n")
                    f.write(f"平均距离: {test_results['mean_distance']:.2f}px\n")
                    f.write(f"平均置信度: {test_results['confidence']:.4f}\n")

                # 保存到 fold_test_results 用于汇总
                if 'fold_test_results' not in locals():
                    fold_test_results = []
                fold_test_results.append(test_results['hit_rate'])

    # 汇总结果
    print("\n" + "=" * 60)
    print("K折交叉验证结果汇总")
    print("=" * 60)

    for i, hr in enumerate(fold_results):
        print(f"Fold {i + 1}: {hr:.2%}")

    mean_hr = np.mean(fold_results)
    std_hr = np.std(fold_results)
    print(f"\n平均 Hit Rate@10: {mean_hr:.2%} ± {std_hr:.2%}")

    # 保存验证集结果
    results_path = output_dir / 'kfold_results.txt'
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"K折交叉验证结果 (K={args.kfold})\n")
        f.write("=" * 40 + "\n")
        f.write("[验证集结果]\n")
        for i, hr in enumerate(fold_results):
            f.write(f"Fold {i + 1}: {hr:.4f}\n")
        f.write(f"\n平均 Hit Rate@10: {mean_hr:.4f} ± {std_hr:.4f}\n")

        # 添加测试集结果
        if 'fold_test_results' in locals() and fold_test_results:
            f.write("\n[测试集结果]\n")
            for i, hr in enumerate(fold_test_results):
                f.write(f"Fold {i + 1}: {hr:.4f}\n")
            mean_test_hr = np.mean(fold_test_results)
            std_test_hr = np.std(fold_test_results)
            f.write(f"\n平均 Hit Rate@5: {mean_test_hr:.4f} ± {std_test_hr:.4f}\n")

    print(f"\n结果已保存到: {results_path}")

    # 打印测试集汇总
    if 'fold_test_results' in locals() and fold_test_results:
        print("\n" + "=" * 60)
        print("测试集评估结果汇总")
        print("=" * 60)
        for i, hr in enumerate(fold_test_results):
            print(f"Fold {i + 1}: {hr:.2%}")
        mean_test_hr = np.mean(fold_test_results)
        std_test_hr = np.std(fold_test_results)
        print(f"\n测试集平均 Hit Rate@5: {mean_test_hr:.2%} ± {std_test_hr:.2%}")


def create_model_and_optimizer(args, device):
    """创建模型、优化器、调度器和混合精度"""
    # 创建模型
    print("\n" + "=" * 50)
    print("创建模型...")
    print("=" * 50)

    if args.use_swin:
        model = GuidanceHeadSwin(
            in_chans=1,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_frame=args.clip_len,
            swin_depth=args.swin_depth,
            swin_heads=args.swin_heads,
            use_spatial_attn=args.use_spatial_attn,
            use_conv3d_mlp=not args.no_conv3d_mlp
        ).to(device)
        print(f"架构: STSF + Swin 3D (depth={args.swin_depth}, heads={args.swin_heads}, conv3d_mlp={not args.no_conv3d_mlp})")
    else:
        model = GuidanceHead(
            in_chans=1,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_frame=args.clip_len
        ).to(device)
        print("架构: STSF + CNN Decoder")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    # 创建损失函数
    if args.loss_type == 'heatmap_focal':
        criterion = HeatmapFocalLoss(alpha=args.focal_alpha, beta=args.focal_beta)
    else:
        criterion = build_loss(args.loss_type)
    print(f"损失函数: {args.loss_type}")

    # 创建优化器
    MUON_LR_MULTIPLIER = 10  # Muon 学习率倍增因子

    if args.optimizer == 'muon_adam':
        # Muon-Adam 混合优化器
        muon_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # 2D 和 4D 参数使用 Muon（矩阵权重）
            if param.ndim in [2, 4] and "relative_position_bias_table" not in name and "dwconv" not in name:
                muon_params.append(param)
            else:
                other_params.append(param)

        muon_lr = args.lr * MUON_LR_MULTIPLIER
        optimizer = [
            Muon(muon_params, lr=muon_lr, momentum=0.95),
            optim.AdamW(other_params, lr=args.lr, weight_decay=args.weight_decay)
        ]
        print(f"优化器: Muon(LR={muon_lr}) + AdamW(LR={args.lr})")
        print(f"  Muon 参数: {sum(p.numel() for p in muon_params):,}")
        print(f"  AdamW 参数: {sum(p.numel() for p in other_params):,}")
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        print(f"优化器: AdamW(LR={args.lr})")

    # 创建学习率调度器
    if isinstance(optimizer, list):
        # 为 Muon 和 AdamW 都创建独立的调度器
        muon_lr = args.lr * MUON_LR_MULTIPLIER
        scheduler = [
            CosineAnnealingLR(optimizer[0], T_max=args.epochs, eta_min=muon_lr * 0.01),
            CosineAnnealingLR(optimizer[1], T_max=args.epochs, eta_min=args.lr * 0.01)
        ]
        print(f"学习率调度: 两个优化器都使用 CosineAnnealing")
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # 混合精度
    scaler = torch.amp.GradScaler('cuda')

    return model, criterion, optimizer, scheduler, scaler


def run_training(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler,
                 device, args, output_dir, writer, fold_idx=None):
    """运行训练循环"""
    best_hit_rate = 0.0
    no_improve = 0

    # 计算评估阈值
    # 原始分辨率 (500x1000) 使用 10px
    # 256x256 分辨率使用 5px（保持相同的目标相对比例）
    if args.threshold_radius is not None:
        threshold_radius = args.threshold_radius
    elif args.img_size is not None:
        # 根据分辨率缩放
        # 原始高度500 -> 256，缩放比例 0.512
        scale = args.img_size[0] / 500.0
        threshold_radius = max(3, int(10 * scale))  # 至少3像素
    else:
        threshold_radius = 10  # 原始分辨率默认值

    print(f"评估阈值: {threshold_radius} 像素")

    fold_prefix = f"[Fold {fold_idx + 1}] " if fold_idx is not None else ""

    # 训练循环
    print("\n" + "=" * 50)
    print(f"{fold_prefix}开始训练...")
    print("=" * 50)

    for epoch in range(args.epochs):
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, args.accum_steps, epoch, args.log_interval
        )

        # 验证
        val_loss = validate(model, val_loader, criterion, device)

        # 评估定位准确率
        hit_rate, mean_distance, mean_confidence = evaluate_localization_accuracy(
            model, val_loader, device, threshold_radius=threshold_radius
        )

        # 更新学习率
        if isinstance(scheduler, list):
            for sch in scheduler:
                sch.step()
        else:
            scheduler.step()

        # 记录日志
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/hit_rate', hit_rate, epoch)
        writer.add_scalar('Metrics/mean_distance', mean_distance, epoch)
        writer.add_scalar('Metrics/mean_confidence', mean_confidence, epoch)
        # 记录学习率（AdamW的学习率）
        if isinstance(scheduler, list):
            writer.add_scalar('LR_muon', scheduler[0].get_last_lr()[0], epoch)
            writer.add_scalar('LR_adamw', scheduler[1].get_last_lr()[0], epoch)
            current_lr = scheduler[1].get_last_lr()[0]
        else:
            writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
            current_lr = scheduler.get_last_lr()[0]

        # 打印结果
        print(f"\n{fold_prefix}Epoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Hit Rate@10: {hit_rate:.2%}")
        print(f"  Mean Distance: {mean_distance:.2f} px")
        print(f"  Mean Confidence: {mean_confidence:.4f}")
        print(f"  LR: {current_lr:.2e}")

        # 保存检查点
        checkpoint_path = output_dir / 'latest.pth'
        if isinstance(optimizer, list):
            optimizer_state = [opt.state_dict() for opt in optimizer]
        else:
            optimizer_state = optimizer.state_dict()

        if isinstance(scheduler, list):
            scheduler_state = [sch.state_dict() for sch in scheduler]
        else:
            scheduler_state = scheduler.state_dict()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': scheduler_state,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'hit_rate': hit_rate,
            'best_hit_rate': best_hit_rate
        }, checkpoint_path)

        # 保存最佳模型
        if hit_rate > best_hit_rate:
            best_hit_rate = hit_rate
            no_improve = 0
            best_path = output_dir / 'best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'hit_rate': hit_rate,
                'mean_distance': mean_distance
            }, best_path)
            print(f"  ✓ 保存最佳模型 (Hit Rate: {hit_rate:.2%})")
        else:
            no_improve += 1

        # 早停
        if no_improve >= args.patience:
            print(f"\n{fold_prefix}早停！验证集 Hit Rate 已 {args.patience} 轮未提升")
            break

        # 达到目标
        if hit_rate >= 0.90:
            print(f"\n{fold_prefix}🎉 达到目标！Hit Rate: {hit_rate:.2%} >= 90%")
            break

    return best_hit_rate


if __name__ == '__main__':
    main()
