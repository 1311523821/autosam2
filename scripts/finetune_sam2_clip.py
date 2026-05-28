#!/usr/bin/env python
"""
SAM2 Clip 训练 — 完整 Memory Attention（含 cross_attn）微调

使用 forked Sam2VideoPredictor，去掉 @torch.inference_mode()，
完整的视频管线（init_state → propagate_in_video）在训练时完全可导。
"""

import os, sys, argparse, json
import torch, numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, '/root/autosam2')
sys.path.insert(0, '/root/e2e')

from models.sam2_clip_trainer import SAM2ClipTrainer, CombinedLoss
from scripts.finetune_sam2_lora import SonarClipDataset, SonarFrameDataset, validate

def parse_args():
    p = argparse.ArgumentParser(description='SAM2 Clip Training (Full Memory Attn)')
    p.add_argument('--sam2-config', default='sam2_hiera_t.yaml')
    p.add_argument('--sam2-ckpt', default='checkpoints/sam2.1_hiera_tiny.pt')
    p.add_argument('--data-root', default='/root/DataBscan')
    p.add_argument('--train-list', default='/root/e2e/train1.txt')
    p.add_argument('--test-list', default='/root/e2e/test1.txt')
    p.add_argument('--output', default='results/phase3_clip_full')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--lora-rank', type=int, default=4)
    p.add_argument('--lora-stages', default='2,3')
    p.add_argument('--val-every', type=int, default=5)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--warmup-epochs', type=int, default=3)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume-from', default=None)
    p.add_argument('--device', default='cuda')
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); np.random.seed(args.seed)

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / 'checkpoints'; ckpt_dir.mkdir(exist_ok=True)

    with open(args.train_list) as f:
        train_folders = [l.strip().replace('\\','/') for l in f if l.strip()]
    with open(args.test_list) as f:
        test_folders = [l.strip().replace('\\','/') for l in f if l.strip()]

    print("=" * 60)
    print("SAM2 Clip Training (Full Memory Attention)")
    print("=" * 60)
    print(f"训练视频: {len(train_folders)}, 验证视频: {len(test_folders)}")

    # DataLoader: clip 模式
    train_ds = SonarClipDataset(train_folders, args.data_root, 'uuv', 1024)
    val_ds = SonarFrameDataset(test_folders, args.data_root, 'uuv', 1024)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"训练 clips: {len(train_ds)}, 验证帧: {len(val_ds)}")

    # 模型
    inject_stages = [int(s) for s in args.lora_stages.split(',')]
    model = SAM2ClipTrainer(args.sam2_config, args.sam2_ckpt, device=args.device,
                            lora_rank=args.lora_rank, inject_stages=inject_stages, finetune_memory='lora')

    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=args.lr, weight_decay=1e-4)

    def warmup_cosine(e):
        if e < args.warmup_epochs: return (e+1)/args.warmup_epochs
        p = (e-args.warmup_epochs)/(args.epochs-args.warmup_epochs)
        return 0.5*(1+np.cos(np.pi*p))*0.99+0.01
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine)

    loss_fn = CombinedLoss(loss_type='dice')
    scaler = torch.amp.GradScaler('cuda')

    history = {'train_loss':[], 'val_loss':[], 'val_iou':[], 'val_dice':[], 'lr':[]}
    best_val_loss, patience_cnt, start_epoch = float('inf'), 0, 1

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=args.device)
        model.model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch',0) + 1
        print(f"恢复至 Epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{args.epochs}\n{'='*60}")
        model.train()
        el, nf = 0.0, 0
        pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch}")
        for images, points, labels, gt in pbar:
            # squeeze DataLoader batch dim: (1, T, 3, H, W) → (T, 3, H, W)
            images = images.squeeze(0).to(args.device)
            points = points.squeeze(0)  # (T, 2)
            gt = gt.squeeze(0)  # (T, 1, H, W)

            r = model.train_clip(images, points, gt, optimizer, loss_fn,
                                 grad_clip=1.0, scaler=scaler)
            if r['valid']:
                el += r['loss']; nf += 1
                pbar.set_postfix({'loss': f"{r['loss']:.4f}"})

        scheduler.step()
        avg_loss = el / max(nf, 1)
        history['train_loss'].append(avg_loss)
        history['lr'].append(scheduler.get_last_lr()[0])
        print(f"平均训练损失: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

        if epoch % args.val_every == 0 or epoch == args.epochs:
            vm = validate(model, val_loader, loss_fn, args.device)
            history['val_loss'].append(vm['loss']); history['val_iou'].append(vm['iou']); history['val_dice'].append(vm['dice'])
            print(f"验证: loss={vm['loss']:.4f}, IoU={vm['iou']:.4f}, Dice={vm['dice']:.4f}")

            if vm['loss'] < best_val_loss:
                best_val_loss = vm['loss']; patience_cnt = 0
                torch.save({'epoch':epoch, 'model_state_dict':model.model.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict(), 'train_loss':avg_loss, 'val_metrics':vm},
                           ckpt_dir/'best_train.pth')
                model.save_for_inference(str(ckpt_dir/'best_inference.pth'), merge_lora=True)
                print(f"✓ 保存最佳: val_loss={vm['loss']:.4f}, IoU={vm['iou']:.4f}")
            elif args.patience > 0:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print("早停!"); break

    json.dump(history, open(out/'training_history.json','w'), indent=2)
    print(f"\n训练完成. 最佳 val_loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()
