#!/usr/bin/env python
"""
SAM 2 LoRA 微调训练脚本 (DataLoader 版本)

使用 PyTorch DataLoader 多进程并行加载数据，替代手动缓存。
支持 LoRA + Mask Decoder + Memory Attention 微调。

用法:
    python scripts/finetune_sam2_lora.py --epochs 50
"""

import os, sys, argparse, json, cv2
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/root/autosam2')
sys.path.insert(0, '/root/e2e')

from models.sam2_finetuner_lora import SAM2LoRAFineTuner, CombinedLoss
from utils.heatmap import get_mask_center

# ============================================================
# 参数
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description='SAM 2 LoRA 微调训练')
    p.add_argument('--sam2-config', default='sam2_hiera_t.yaml')
    p.add_argument('--sam2-ckpt', default='checkpoints/sam2.1_hiera_tiny.pt')
    p.add_argument('--data-root', default='/root/DataBscan')
    p.add_argument('--train-list', default='/root/e2e/train1.txt')
    p.add_argument('--test-list', default='/root/e2e/test1.txt')
    p.add_argument('--output', default='results/phase3_sam2_lora')
    p.add_argument('--target-label', default='uuv')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--lora-rank', type=int, default=4)
    p.add_argument('--lora-alpha', type=int, default=8)
    p.add_argument('--lora-stages', default='2,3')
    p.add_argument('--optimizer', default='adamw', choices=['adamw','muon_adam'])
    p.add_argument('--loss-type', default='dice', choices=['dice','tversky'])
    p.add_argument('--image-size', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--val-every', type=int, default=5)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--warmup-epochs', type=int, default=3)
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--no-amp', action='store_false', dest='amp')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--finetune-memory', default='none', choices=['none','full','lora'])
    p.add_argument('--resume-from', default=None)
    p.add_argument('--device', default='cuda')
    return p.parse_args()

# ============================================================
# 数据增强 (GPU)
# ============================================================
def apply_augmentation(batch):
    if np.random.random() < 0.5:
        for f in batch:
            f['image'] = torch.flip(f['image'], dims=[-1])
            f['gt_mask'] = torch.flip(f['gt_mask'], dims=[-1])
            W = f['image'].shape[-1]
            f['point'][..., 0] = W - f['point'][..., 0]

    if np.random.random() < 0.3:
        batch.reverse()

    if np.random.random() < 0.5:
        brightness = 0.85 + np.random.random() * 0.3
        contrast = 0.9 + np.random.random() * 0.2
        for f in batch:
            f['image'] = torch.clamp((f['image'] * brightness + (1-brightness)*0.5) * contrast, -3, 3)

    if np.random.random() < 0.3 and len(batch) > 0:
        W_img = batch[0]['image'].shape[-1]
        H_img = batch[0]['image'].shape[-2]
        sx = int(0.05 * np.random.randn() * W_img)
        sy = int(0.05 * np.random.randn() * H_img)
        for f in batch:
            f['image'] = torch.roll(f['image'], shifts=(sy, sx), dims=(-2, -1))
            f['gt_mask'] = torch.roll(f['gt_mask'], shifts=(sy, sx), dims=(-2, -1))
            f['point'][..., 0] += sx
            f['point'][..., 1] += sy
    return batch

# ============================================================
# Dataset
# ============================================================
class SonarFrameDataset(Dataset):
    IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMG_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, folders, data_root, target_label, image_size):
        self.image_size = image_size
        self.target_label = target_label
        self.samples = []
        for folder in tqdm(folders, desc="构建数据集"):
            d = os.path.join(data_root, folder)
            if not os.path.isdir(d): continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(('.png','.jpg','.jpeg')):
                    j = os.path.join(d, os.path.splitext(f)[0] + '.json')
                    if os.path.exists(j):
                        self.samples.append((d, f, j))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        folder, name, json_path = self.samples[idx]
        img = cv2.imread(os.path.join(folder, name), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape

        mask = np.zeros((h,w), dtype=np.uint8)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for s in data.get('shapes',[]):
            if s['label'] == self.target_label:
                cv2.fillPoly(mask, [np.array(s['points'],np.float32).astype(np.int32)], 1)

        if mask.sum() == 0:
            return self[(idx + 1) % len(self)]

        center = get_mask_center(torch.from_numpy(mask))
        if center is None:
            return self[(idx + 1) % len(self)]

        sx, sy = self.image_size / w, self.image_size / h
        img_r = cv2.resize(img, (self.image_size, self.image_size))
        mask_r = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        img_t = torch.from_numpy(img_r).float() / 255.0
        img_t = img_t.unsqueeze(0).expand(3, -1, -1)
        img_t = (img_t - self.IMG_MEAN) / self.IMG_STD

        return (
            img_t,  # (3, H, W)
            np.array([center[0]*sx, center[1]*sy], dtype=np.float32),  # (2,)
            np.array([1], dtype=np.int64),  # (1,)
            torch.from_numpy(mask_r).float().unsqueeze(0),  # (1, H, W)
        )

# ============================================================
# 验证 (简单版，只看 loss)
# ============================================================
def compute_iou(pred, gt, t=0.5):
    b = (pred > t).float()
    inter = (b * gt).sum()
    union = b.sum() + gt.sum() - inter
    return (inter / union).item() if union > 0 else 0.0

def compute_dice(pred, gt, t=0.5):
    b = (pred > t).float()
    inter = (b * gt).sum()
    total = b.sum() + gt.sum()
    return (2*inter / total).item() if total > 0 else 0.0

def validate(model, loader, loss_fn, device):
    model.eval()
    tl, ti, td, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for images, pts_np, lbls_np, gt in loader:
            images = images.to(device)
            pts = pts_np.unsqueeze(1).to(device)
            lbls = lbls_np.to(device)
            gt = gt.to(device)

            m, _ = model.forward_single_frame(images, pts, lbls)
            pm = F.interpolate(m, size=gt.shape[-2:], mode='bilinear', align_corners=False)
            ps = torch.sigmoid(pm)

            tl += loss_fn(pm, gt).item()
            for i in range(ps.shape[0]):
                ti += compute_iou(ps[i], gt[i])
                td += compute_dice(ps[i], gt[i])
            n += ps.shape[0]

    return {'loss': tl/max(n,1), 'iou': ti/max(n,1), 'dice': td/max(n,1)}

# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    # 列表
    with open(args.train_list) as f:
        train_folders = [l.strip().replace('\\','/') for l in f if l.strip()]
    with open(args.test_list) as f:
        test_folders = [l.strip().replace('\\','/') for l in f if l.strip()]

    print("=" * 60)
    print("SAM 2 LoRA 微调训练 (DataLoader)")
    print("=" * 60)
    print(f"训练视频: {len(train_folders)}, 验证视频: {len(test_folders)}")
    print(f"LR: {args.lr}, LoRA rank: {args.lora_rank}, stages: {args.lora_stages}")
    print(f"Loss: {args.loss_type}, optimizer: {args.optimizer}")
    print(f"Batch: {args.batch_size}, grad_accum: {args.grad_accum}")

    # DataLoader
    train_ds = SonarFrameDataset(train_folders, args.data_root, args.target_label, args.image_size)
    val_ds   = SonarFrameDataset(test_folders, args.data_root, args.target_label, args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    print(f"训练帧: {len(train_ds)}, 验证帧: {len(val_ds)}")

    # 模型
    inject_stages = [int(s) for s in args.lora_stages.split(',')]
    model = SAM2LoRAFineTuner(args.sam2_config, args.sam2_ckpt, device=args.device,
                              lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                              inject_stages=inject_stages, finetune_memory=args.finetune_memory)

    # 优化器
    use_dual = False
    if args.optimizer == 'muon_adam':
        from utils.muon import Muon
        muon_p, adam_p = [], []
        for name, param in model.model.named_parameters():
            if not param.requires_grad: continue
            if param.ndim in [2,4] and 'bias' not in name:
                muon_p.append(param)
            else:
                adam_p.append(param)
        optimizer = Muon(muon_p, lr=args.lr*10, momentum=0.95) if muon_p else None
        adam_opt  = torch.optim.AdamW(adam_p, lr=args.lr, weight_decay=args.weight_decay) if adam_p else None
        use_dual = True
    else:
        optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
        adam_opt = None

    # 学习率
    def warmup_cosine(e):
        if e < args.warmup_epochs: return (e+1) / args.warmup_epochs
        p = (e - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
        return 0.5*(1+np.cos(np.pi*p))*0.99 + 0.01

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine)
    sched_adam = torch.optim.lr_scheduler.LambdaLR(adam_opt, warmup_cosine) if adam_opt else None

    # Loss
    loss_fn = CombinedLoss(loss_type=args.loss_type)
    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    GRAD_CLIP = 1.0

    history = {'train_loss':[], 'val_loss':[], 'val_iou':[], 'val_dice':[], 'lr':[]}
    best_val_loss = float('inf')
    patience_cnt = 0
    start_epoch = 1

    # 恢复
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=args.device)
        model.model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch',0) + 1
        best_val_loss = ckpt.get('val_metrics',{}).get('loss', float('inf'))
        print(f"恢复至 Epoch {start_epoch}")

    # 训练
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{args.epochs}\n{'='*60}")
        model.train()
        el, nf, acc = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch}")
        for images, pts_np, lbls_np, gt in pbar:
            images = images.to(args.device, non_blocking=True)
            pts = pts_np.unsqueeze(1).to(args.device, non_blocking=True)
            lbls = lbls_np.to(args.device, non_blocking=True)
            gt = gt.to(args.device, non_blocking=True)

            # GPU 增强
            gf = [{'image':images[i], 'gt_mask':gt[i], 'point':pts[i]} for i in range(len(images))]
            gf = apply_augmentation(gf)
            images = torch.stack([x['image'] for x in gf])
            pts    = torch.stack([x['point'] for x in gf])

            if use_dual and acc == 0:
                optimizer.zero_grad()
                if adam_opt: adam_opt.zero_grad()

            r = model.train_step(images, pts, lbls, gt, optimizer, loss_fn,
                                 grad_clip=GRAD_CLIP, scaler=scaler,
                                 second_optimizer=adam_opt if use_dual else None,
                                 skip_step=(acc+1 < args.grad_accum))

            acc += 1
            if acc >= args.grad_accum: acc = 0
            if r['valid']:
                el += r['loss']; nf += 1
                pbar.set_postfix({'loss': f"{r['loss']:.4f}"})

        scheduler.step()
        if sched_adam: sched_adam.step()
        avg_loss = el / max(nf, 1)
        history['train_loss'].append(avg_loss)
        history['lr'].append(scheduler.get_last_lr()[0])
        print(f"平均训练损失: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

        # 验证
        do_val = (epoch % args.val_every == 0) or (epoch == args.epochs)
        if do_val:
            vm = validate(model, val_loader, loss_fn, args.device)
            history['val_loss'].append(vm['loss'])
            history['val_iou'].append(vm['iou'])
            history['val_dice'].append(vm['dice'])
            print(f"验证: loss={vm['loss']:.4f}, IoU={vm['iou']:.4f}, Dice={vm['dice']:.4f}")

            if vm['loss'] < best_val_loss:
                best_val_loss = vm['loss']
                patience_cnt = 0
                torch.save({'epoch':epoch, 'model_state_dict':model.model.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict(),
                            'train_loss':avg_loss, 'val_metrics':vm, 'lora_rank':args.lora_rank},
                           ckpt_dir/'best_train.pth')
                model.save_for_inference(str(ckpt_dir/'best_inference.pth'), merge_lora=True)
                print(f"✓ 保存最佳: val_loss={vm['loss']:.4f}, IoU={vm['iou']:.4f}")
            elif args.patience > 0:
                patience_cnt += 1
                print(f"  未改善 ({patience_cnt}/{args.patience})")
                if patience_cnt >= args.patience:
                    print(f"早停！")
                    break

        if epoch % args.save_every == 0:
            torch.save({'epoch':epoch, 'model_state_dict':model.model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict()},
                       ckpt_dir/f'epoch_{epoch}_train.pth')

    # 结束
    json.dump(history, open(output_dir/'training_history.json','w'), indent=2)
    torch.save({'epoch':args.epochs, 'model_state_dict':model.model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict()}, ckpt_dir/'final.pth')
    print(f"\n训练完成。最佳 val_loss: {best_val_loss:.4f}")
    print(f"输出: {output_dir}")

if __name__ == '__main__':
    main()
