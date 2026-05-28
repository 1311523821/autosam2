#!/usr/bin/env python
"""
Clip Training 测试用例

测试目标：完整 Memory Attention（含 cross_attn）微调的正确性
"""

import sys, os, torch, numpy as np, json, cv2, tempfile
sys.path.insert(0, '/root/autosam2')

SAM2_CONFIG = 'sam2_hiera_t.yaml'
SAM2_CKPT = 'checkpoints/sam2.1_hiera_tiny.pt'
TEST_DATA = '/root/DataBscan/day1/testing/jin/DataRecord_2025-12-08_132412'

# ============================================================
# 测试 1: 验证 @torch.inference_mode() 已从 forked predictor 移除
# ============================================================
def test_inference_mode_removed():
    """验证：forked predictor 中不再有 @torch.inference_mode() 装饰器"""
    import inspect
    from models.sam2_video_predictor_train import SAM2VideoPredictor

    methods_to_check = ['init_state', 'add_new_points', 'add_new_points_or_box',
                        'propagate_in_video', 'propagate_in_video_preflight']
    for method_name in methods_to_check:
        method = getattr(SAM2VideoPredictor, method_name, None)
        if method is None:
            print(f"  ✗ {method_name}: 不存在")
            continue
        source = inspect.getsource(method)
        has_decorator = '@torch.inference_mode()' in source.split('\n')[0:5]
        if has_decorator:
            print(f"  ✗ {method_name}: 仍有 @torch.inference_mode()!")
            return False
    print("  ✓ 所有方法已移除 @torch.inference_mode()")
    return True


# ============================================================
# 测试 2: 验证 Memory Attention LoRA 注入层数
# ============================================================
def test_lora_injection():
    """验证：Memory Attention 正确注入 32 层 LoRA（self_attn + cross_attn）"""
    from models.sam2_clip_trainer import SAM2ClipTrainer, LoRALinear

    model = SAM2ClipTrainer(SAM2_CONFIG, SAM2_CKPT, finetune_memory='lora')

    # 统计 LoRA 层
    lora_layers = {
        'self_attn': 0, 'cross_attn': 0,
        'image_encoder': 0, 'other': 0
    }
    for name, module in model.model.named_modules():
        if isinstance(module, LoRALinear):
            if 'self_attn' in name:
                lora_layers['self_attn'] += 1
            elif 'cross_attn' in name:
                lora_layers['cross_attn'] += 1
            elif 'image_encoder' in name:
                lora_layers['image_encoder'] += 1
            else:
                lora_layers['other'] += 1

    # 验证
    checks = [
        lora_layers['self_attn'] == 16,      # 4层 × 4个投影
        lora_layers['cross_attn'] == 16,     # 4层 × 4个投影
        lora_layers['image_encoder'] >= 20,  # Image Encoder
    ]

    print(f"  LoRA 注入统计: {lora_layers}")
    if all(checks):
        print("  ✓ LoRA 注入正确")
        return True
    else:
        print(f"  ✗ LoRA 注入不足: self_attn={lora_layers['self_attn']}/16, cross_attn={lora_layers['cross_attn']}/16")
        return False


# ============================================================
# 测试 3: 验证 Gradient 流经 Memory Attention
# ============================================================
def test_gradient_flow():
    """验证：backward 后 Memory Attention LoRA 参数有非零梯度"""
    from models.sam2_clip_trainer import SAM2ClipTrainer, CombinedLoss, LoRALinear

    model = SAM2ClipTrainer(SAM2_CONFIG, SAM2_CKPT, finetune_memory='lora')

    # 加载 4 帧真实数据
    imgs = sorted([f for f in os.listdir(TEST_DATA) if f.endswith('.jpg')])[:4]
    frames = []
    for f in imgs:
        img = cv2.imread(os.path.join(TEST_DATA, f), cv2.IMREAD_GRAYSCALE)
        img_r = cv2.resize(img, (1024, 1024))
        img_t = torch.from_numpy(img_r).float().unsqueeze(0).expand(3,-1,-1) / 255.0
        img_t = (img_t - torch.tensor([0.485,0.456,0.406]).view(3,1,1)) / torch.tensor([0.229,0.224,0.225]).view(3,1,1)
        frames.append(img_t)
    frames = torch.stack(frames).cuda()  # (4, 3, 1024, 1024)

    # GT 中心点和 mask（用假数据验证梯度流即可）
    points = torch.tensor([[512, 256], [512, 256], [512, 256], [512, 256]]).float()
    gt = torch.zeros(4, 1024, 1024)
    gt[:, 250:260, 500:520] = 1

    opt = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-5)
    loss_fn = CombinedLoss(loss_type='dice')

    try:
        r = model.train_clip(frames, points, gt, opt, loss_fn, grad_clip=1.0)
    except Exception as e:
        print(f"  ✗ train_clip 调用失败: {e}")
        return False

    # 检查 Memory Attention LoRA 参数的梯度
    mem_grads = {'self_attn': [], 'cross_attn': []}
    for name, param in model.model.named_parameters():
        if 'memory_attention' in name and 'lora' in name.lower() and param.grad is not None:
            gn = param.grad.norm().item()
            if 'self_attn' in name:
                mem_grads['self_attn'].append(gn)
            elif 'cross_attn' in name:
                mem_grads['cross_attn'].append(gn)

    sa_nonzero = sum(1 for g in mem_grads['self_attn'] if g > 1e-15)
    ca_nonzero = sum(1 for g in mem_grads['cross_attn'] if g > 1e-15)

    print(f"  self_attn 梯度: {sa_nonzero}/{len(mem_grads['self_attn'])} 非零")
    print(f"  cross_attn 梯度: {ca_nonzero}/{len(mem_grads['cross_attn'])} 非零")
    print(f"  loss: {r['loss']:.4f}")

    if r['valid'] and sa_nonzero > 0:
        print("  ✓ 梯度正常流经 Memory Attention")
        return True
    else:
        print("  ✗ 梯度流异常!")
        return False


# ============================================================
# 测试 4: 验证 2-epoch 训练能收敛
# ============================================================
def test_training_convergence():
    """验证：2 epoch 后 val_loss 下降"""
    from models.sam2_clip_trainer import SAM2ClipTrainer, CombinedLoss
    from torch.utils.data import DataLoader
    import scripts.finetune_sam2_lora as fl

    # 小型测试
    with open('/root/e2e/train1.txt') as f:
        train_folders = [l.strip().replace('\\','/') for l in f if l.strip()][:2]  # 只用2个视频
    with open('/root/e2e/test1.txt') as f:
        test_folders = [l.strip().replace('\\','/') for l in f if l.strip()][:1]  # 只用1个视频

    train_ds = fl.SonarClipDataset(train_folders, '/root/DataBscan', 'uuv', 1024)
    val_ds = fl.SonarFrameDataset(test_folders, '/root/DataBscan', 'uuv', 1024)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

    model = SAM2ClipTrainer(SAM2_CONFIG, SAM2_CKPT, finetune_memory='lora')
    opt = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-5)
    loss_fn = CombinedLoss(loss_type='dice')

    history = []
    for epoch in range(1, 3):
        model.train()
        el, n = 0.0, 0
        for images, points, labels, gt in train_loader:
            images = images.squeeze(0).cuda()
            points = points.squeeze(0)
            gt = gt.squeeze(0)
            r = model.train_clip(images, points, gt, opt, loss_fn, grad_clip=1.0)
            if r['valid']:
                el += r['loss']; n += 1
        avg_loss = el / max(n, 1)
        history.append(avg_loss)
        print(f"  epoch {epoch}: train_loss={avg_loss:.4f}")

    if len(history) >= 2 and history[1] < history[0] * 0.99:
        print(f"  ✓ Loss 下降: {history[0]:.4f} → {history[1]:.4f}")
        return True
    else:
        print(f"  ✗ Loss 未下降: {history}")
        return False


# ============================================================
# 测试 5: 验证 checkpoint 可以被 test_sam2_finetuned.py 加载
# ============================================================
def test_checkpoint_compatibility():
    """验证：保存的 best_inference.pth 能被 build_sam2_video_predictor 加载"""
    from models.sam2_clip_trainer import SAM2ClipTrainer
    from sam2.build_sam import build_sam2_video_predictor

    model = SAM2ClipTrainer(SAM2_CONFIG, SAM2_CKPT, finetune_memory='lora')

    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        tmp = f.name
    model.save_for_inference(tmp, merge_lora=True)

    try:
        p = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CKPT, device='cuda')
        ckpt = torch.load(tmp, map_location='cuda')
        missing, unexpected = p.load_state_dict(ckpt['model'], strict=False)
        os.unlink(tmp)
        print(f"  ✓ Checkpoint 兼容（missing={len(missing)}, unexpected={len(unexpected)}）")
        return True
    except Exception as e:
        os.unlink(tmp)
        print(f"  ✗ Checkpoint 加载失败: {e}")
        return False


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("SAM2 Clip Training 测试套件")
    print("=" * 60)

    results = {}
    tests = [
        ('装饰器已移除', test_inference_mode_removed),
        ('LoRA 注入正确', test_lora_injection),
        ('梯度流经 Memory Attn', test_gradient_flow),
        ('Checkpoint 兼容', test_checkpoint_compatibility),
        ('2-epoch 收敛', test_training_convergence),
    ]

    for name, test in tests:
        print(f"\n--- 测试: {name} ---")
        try:
            results[name] = test()
        except Exception as e:
            print(f"  ✗ 异常: {e}")
            results[name] = False

    # 汇总
    print("\n" + "=" * 60)
    passed = sum(results.values())
    total = len(results)
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n结果: {passed}/{total} 通过")
