#!/usr/bin/env python
"""
SAM2 LoRA 训练器测试

验证:
1. LoRA 模块注入正确
2. 训练器构建成功
3. Cross-attention 梯度流通
4. Single-frame 前向正常
5. Checkpoint 保存/加载正确
"""

import sys
import os
import torch
import numpy as np
import tempfile

sys.path.insert(0, '/root/autosam2')

SAM2_CONFIG = 'sam2_hiera_t.yaml'
SAM2_CKPT = 'checkpoints/sam2.1_hiera_tiny.pt'
TEST_DATA = '/root/DataBscan/day1/testing/jin/DataRecord_2025-12-08_132412'


def test_lora_module():
    """测试 LoRA 模块基本功能"""
    from models.lora import LoRALinear, inject_lora, count_lora_params

    # 测试 LoRALinear
    linear = torch.nn.Linear(64, 128)
    lora = LoRALinear(linear, r=4, alpha=8)

    # 测试 forward
    x = torch.randn(2, 64)
    out = lora(x)
    assert out.shape == (2, 128), f"Shape mismatch: {out.shape}"

    # 测试梯度
    loss = out.sum()
    loss.backward()
    assert lora.lora_A.grad is not None, "lora_A 无梯度"
    assert lora.lora_B.grad is not None, "lora_B 无梯度"
    assert linear.weight.grad is None, "原始权重应该有梯度"  # 原始权重 forward 仍有 grad

    # 测试 inject_lora
    class TestMod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.Sequential(
                torch.nn.Linear(64, 64),
                torch.nn.Linear(64, 128),
            )
            self.q_proj = torch.nn.Linear(128, 256)
            self.k_proj = torch.nn.Linear(64, 256)

    m = TestMod()
    n = inject_lora(m, ['q_proj', 'k_proj'], r=4, alpha=8, verbose=False)
    assert n == 2, f"应注入 2 层，实际 {n}"
    assert isinstance(m.q_proj, LoRALinear)
    assert isinstance(m.k_proj, LoRALinear)

    # 验证 stat
    stats = count_lora_params(m)
    assert stats['total_lora'] > 0, "应有 LoRA 参数"

    print("  ✓ LoRA 模块测试通过")


def test_trainer_build_and_forward():
    """测试训练器构建和单帧前向"""
    from models.sam2_lora_trainer import SAM2LoRATrainer

    trainer = SAM2LoRATrainer(
        sam2_config=SAM2_CONFIG,
        sam2_checkpoint=SAM2_CKPT,
        lora_config={'targets': ['cross_attn'], 'r': 4, 'alpha': 8},
        train_mask_decoder=True,
        device='cuda',
    )

    # 测试单帧前向
    image = torch.randn(1, 3, 1024, 1024).cuda()
    pts = torch.tensor([[[512, 512]]]).float().cuda()
    lbls = torch.tensor([[1]]).int().cuda()

    masks, ious = trainer.forward_single_frame(image, pts, lbls)
    assert masks.shape == (1, 1, 256, 256), f"Mask shape: {masks.shape}"
    assert ious.shape == (1, 1), f"IoU shape: {ious.shape}"

    print("  ✓ 训练器构建和单帧前向通过")


def test_cross_attn_gradient_flow():
    """测试 cross-attention LoRA 梯度流通"""
    from models.sam2_lora_trainer import SAM2LoRATrainer, CombinedLoss
    from models.lora import get_lora_modules

    trainer = SAM2LoRATrainer(
        sam2_config=SAM2_CONFIG,
        sam2_checkpoint=SAM2_CKPT,
        lora_config={'targets': ['cross_attn'], 'r': 4, 'alpha': 8},
        train_mask_decoder=True,
        device='cuda',
    )

    # 构造 4 帧假数据
    T = 4
    images = torch.randn(T, 3, 1024, 1024).cuda()
    points = torch.tensor([[512, 512], [512, 512], [512, 512], [512, 512]]).float()
    gt = torch.zeros(T, 1024, 1024)
    gt[:, 490:530, 490:530] = 1

    opt = torch.optim.AdamW(trainer.get_trainable_params(), lr=1e-5)
    loss_fn = CombinedLoss(seg_type='tversky')

    result = trainer.train_step(images, points, gt, opt, loss_fn, grad_clip=1.0)
    assert result['loss'] > 0, "Loss 应为正值"

    # 检查 cross-attn LoRA 梯度
    lora_mods = get_lora_modules(trainer.model)
    nonzero = 0
    for m in lora_mods:
        if m.lora_A.grad is not None and m.lora_A.grad.norm() > 1e-15:
            nonzero += 1
        if m.lora_B.grad is not None and m.lora_B.grad.norm() > 1e-15:
            nonzero += 1

    print(f"  Cross-attn LoRA 梯度: {nonzero}/{len(lora_mods) * 2} 非零")
    print(f"  Loss: {result['loss']:.4f}")
    assert nonzero > 0, "Cross-attn LoRA 无梯度！"

    print("  ✓ Cross-attention 梯度流通测试通过")


def test_checkpoint_save_load():
    """测试 checkpoint 保存和加载"""
    from models.sam2_lora_trainer import SAM2LoRATrainer
    from models.lora import get_lora_modules

    trainer = SAM2LoRATrainer(
        sam2_config=SAM2_CONFIG,
        sam2_checkpoint=SAM2_CKPT,
        lora_config={'targets': ['cross_attn'], 'r': 4, 'alpha': 8},
        train_mask_decoder=True,
        device='cuda',
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # 保存推理 checkpoint
        inf_path = os.path.join(tmpdir, 'inference.pth')
        trainer.save_checkpoint(inf_path, merge_lora=True)
        ckpt = torch.load(inf_path, map_location='cuda')
        assert 'model' in ckpt, "推理 checkpoint 应有 'model' key"
        assert 'lora_A' not in str(ckpt['model'].keys()), "合并后不应有 lora 参数"

        # 保存训练 checkpoint
        train_path = os.path.join(tmpdir, 'train.pth')
        trainer.save_checkpoint(train_path, merge_lora=False,
                                extra={'epoch': 1, 'optimizer_state_dict': {}})
        ckpt2 = torch.load(train_path, map_location='cuda')
        assert 'model_state_dict' in ckpt2, "训练 checkpoint 应有 model_state_dict"

    print("  ✓ Checkpoint 保存/加载测试通过")


def test_lora_config_switch():
    """测试不同 LoRA 配置均可正常构建"""
    from models.sam2_lora_trainer import SAM2LoRATrainer

    configs = [
        {'targets': ['cross_attn'], 'r': 4, 'alpha': 8},
        {'targets': ['self_attn'], 'r': 4, 'alpha': 8},
        {'targets': ['memory_attn'], 'r': 4, 'alpha': 8},
        {'targets': ['cross_attn', 'image_encoder'], 'r': 4, 'alpha': 8},
    ]

    for cfg in configs:
        trainer = SAM2LoRATrainer(
            sam2_config=SAM2_CONFIG,
            sam2_checkpoint=SAM2_CKPT,
            lora_config=cfg,
            train_mask_decoder=True,
            device='cuda',
        )
        params = trainer.get_trainable_params()
        assert len(params) > 0, f"Config {cfg['targets']}: 无可训练参数"
        print(f"  {cfg['targets']}: {len(params)} 个参数组")

    print("  ✓ LoRA 配置切换测试通过")


# ============================================================
# 测试 6: 验证 validate() 使用官方 predictor + 仅 frame 0 GT 点
# ============================================================
def test_validate_uses_predictor():
    """验证: validate() 走 predictor.propagate_in_video()，仅 frame 0 注入点"""
    import inspect, re
    from scripts.train_sam2_lora import validate, _build_val_predictor

    # 检查源码是否包含关键的 predictor API 调用
    validate_src = inspect.getsource(validate)
    build_src = inspect.getsource(_build_val_predictor)

    checks = [
        ('build_sam2_video_predictor' in build_src, '使用官方 build_sam2_video_predictor'),
        ('load_state_dict' in build_src, '通过 load_state_dict 注入 LoRA 权重'),
        ('predictor.add_new_points_or_box' in validate_src, '调用 add_new_points_or_box'),
        ('predictor.propagate_in_video' in validate_src, '调用 propagate_in_video'),
        ('frame_idx=0' in validate_src, '仅 frame 0 注入点'),
    ]

    all_pass = True
    for ok, desc in checks:
        status = '✓' if ok else '✗'
        if not ok:
            all_pass = False
        print(f"  {status} {desc}")

    assert all_pass, "validate() 实现与设计要求不符"
    print("  ✓ validate() 使用官方 predictor API")


# ============================================================
# 测试 7: 验证 predictor 输出 mask 形状与 GT 一致
# ============================================================
def test_predictor_output_shape():
    """验证: predictor 传播输出的 mask 经过 sigmoid+squeeze 后与 GT 形状一致"""
    import json, cv2
    from models.sam2_lora_trainer import SAM2LoRATrainer
    from sam2.build_sam import build_sam2_video_predictor
    from utils.heatmap import get_mask_center

    trainer = SAM2LoRATrainer(
        sam2_config=SAM2_CONFIG, sam2_checkpoint=SAM2_CKPT,
        lora_config={'targets': ['cross_attn'], 'r': 4, 'alpha': 8},
        train_mask_decoder=True, device='cuda',
    )

    # 加载首个测试视频的第一段 clip
    with open('/root/e2e/test1.txt') as f:
        test_folders = [l.strip().replace('\\', '/') for l in f if l.strip()][:1]

    folder = test_folders[0]
    d = os.path.join('/root/DataBscan', folder)
    files = sorted([f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    valid = [f for f in files if os.path.exists(os.path.join(d, os.path.splitext(f)[0] + '.json'))][:4]

    raw_frames, gt_masks, gt_center0 = [], {}, None
    for fi, fname in enumerate(valid):
        img = cv2.imread(os.path.join(d, fname), cv2.IMREAD_GRAYSCALE)
        raw_frames.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
        jp = os.path.join(d, os.path.splitext(fname)[0] + '.json')
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        with open(jp, 'r') as jf:
            data = json.load(jf)
        for s in data.get('shapes', []):
            if s.get('label') == 'uuv':
                cv2.fillPoly(mask, [np.array(s['points'], dtype=np.float32).astype(np.int32)], 1)
        gt_masks[fi] = mask
        if fi == 0 and mask.sum() > 0:
            c = get_mask_center(torch.from_numpy(mask))
            gt_center0 = (float(c[0]), float(c[1]))

    if gt_center0 is None:
        print("  ⚠ 跳过: 首帧无目标")
        return

    # 构建 predictor 并注入 LoRA 合并权重（与 validate/_build_val_predictor 一致）
    predictor = build_sam2_video_predictor(trainer.sam2_config, device='cuda')
    merged_sd = trainer._build_merged_state_dict()
    predictor.load_state_dict(merged_sd, strict=False)

    # 手动初始化（与 validate 一致）
    img_mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(3, 1, 1)
    img_std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(3, 1, 1)
    img_tensors = []
    for f in raw_frames:
        f_r = cv2.resize(f, (1024, 1024))
        t = torch.from_numpy(f_r).permute(2, 0, 1).float().cuda() / 255.0
        t = (t - img_mean) / img_std
        img_tensors.append(t)

    is_state = {
        'images': img_tensors, 'num_frames': len(img_tensors),
        'video_height': raw_frames[0].shape[0], 'video_width': raw_frames[0].shape[1],
        'device': torch.device('cuda'), 'storage_device': torch.device('cpu'),
        'point_inputs_per_obj': {}, 'mask_inputs_per_obj': {},
        'cached_features': {}, 'constants': {},
        'obj_id_to_idx': {}, 'obj_idx_to_id': {}, 'obj_ids': [],
        'output_dict_per_obj': {}, 'temp_output_dict_per_obj': {},
        'frames_tracked_per_obj': {},
        'offload_video_to_cpu': True, 'offload_state_to_cpu': True,
    }

    predictor.add_new_points_or_box(
        inference_state=is_state, frame_idx=0, obj_id=1,
        points=np.array([gt_center0], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )

    shape_checks = []
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(is_state):
        if out_frame_idx in gt_masks:
            pred_logit = out_mask_logits[0].cpu().numpy()
            pred_prob = 1.0 / (1.0 + np.exp(-pred_logit))
            pred_mask = (pred_prob > 0.5).astype(np.uint8).squeeze(0)
            gt = gt_masks[out_frame_idx]
            ok = pred_mask.shape == gt.shape
            shape_checks.append(ok)
            if not ok:
                print(f"  ✗ Frame {out_frame_idx}: pred={pred_mask.shape}, gt={gt.shape}")

    all_ok = all(shape_checks)
    print(f"  形状检查: {sum(shape_checks)}/{len(shape_checks)} 通过")
    assert all_ok, "预测 mask 形状与 GT 不匹配"
    print("  ✓ Predictor 输出形状与 GT 一致")


# ============================================================
# 测试 8: 验证 validate() 返回完整 NUDT 指标
# ============================================================
def test_validate_returns_nudt():
    """验证: validate() 返回值包含所有 NUDT 指标，且在合理范围内"""
    from models.sam2_lora_trainer import SAM2LoRATrainer, CombinedLoss
    from scripts.train_sam2_lora import validate

    trainer = SAM2LoRATrainer(
        sam2_config=SAM2_CONFIG, sam2_checkpoint=SAM2_CKPT,
        lora_config={'targets': ['cross_attn'], 'r': 4, 'alpha': 8},
        train_mask_decoder=True, device='cuda',
    )

    with open('/root/e2e/test1.txt') as f:
        test_folders = [l.strip().replace('\\', '/') for l in f if l.strip()][:1]

    vm = validate(trainer, test_folders, '/root/DataBscan', 1024, 4,
                  CombinedLoss(seg_type='tversky'), 'cuda', trainer.sam2_config)

    required_keys = ['loss', 'iou', 'niou', 'dice', 'pd', 'fa']
    for k in required_keys:
        assert k in vm, f"缺少 key: {k}"
        assert 0.0 <= vm[k] <= 1.0, f"{k}={vm[k]} 不在 [0, 1]"

    print(f"  IoU={vm['iou']:.4f}, nIoU={vm['niou']:.4f}, Pd={vm['pd']:.3f}, Fa={vm['fa']:.5f}")
    print("  ✓ validate() 返回完整 NUDT 指标")


# ============================================================
# 测试 9: 验证 test_sam2_lora.py 的 baseline/non-baseline 构建
# ============================================================
def test_test_script_predictor_init():
    """验证: test_sam2_lora.py 的 predictor 初始化不抛异常"""
    from sam2.build_sam import build_sam2_video_predictor
    from models.sam2_lora_trainer import SAM2LoRATrainer

    trainer = SAM2LoRATrainer(
        sam2_config=SAM2_CONFIG, sam2_checkpoint=SAM2_CKPT,
        lora_config={'targets': ['cross_attn'], 'r': 4, 'alpha': 8},
        train_mask_decoder=True, device='cuda',
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        inf_path = os.path.join(tmpdir, 'best_inference.pth')
        trainer.save_checkpoint(inf_path, merge_lora=True)

        # 加载合并后的 checkpoint 并验证 predictor 可构建
        ckpt = torch.load(inf_path, map_location='cuda')
        sd = ckpt['model']

        predictor = build_sam2_video_predictor(SAM2_CONFIG, device='cuda')
        # strict=False 处理 SAM2 版本差异导致的额外 key
        missing, unexpected = predictor.load_state_dict(sd, strict=False)
        assert predictor is not None
        # missing 应为 0（所有权重都被加载），unexpected 应被记录但不阻塞
        assert len(missing) == 0, f"{len(missing)} 个 key 未能加载"
        print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)} (strict=False)")

    print("  ✓ 合并后的 checkpoint 可被 predictor 加载")


if __name__ == '__main__':
    print("=" * 60)
    print("SAM2 LoRA Trainer 测试")
    print("=" * 60)

    tests = [
        ('LoRA 模块', test_lora_module),
        ('构建和单帧前向', test_trainer_build_and_forward),
        ('Cross-attn 梯度流通', test_cross_attn_gradient_flow),
        ('Checkpoint 保存/加载', test_checkpoint_save_load),
        ('LoRA 配置切换', test_lora_config_switch),
        ('validate() 使用官方 predictor', test_validate_uses_predictor),
        ('Predictor 输出形状与 GT 一致', test_predictor_output_shape),
        ('validate() 返回 NUDT 指标', test_validate_returns_nudt),
        ('test_sam2_lora predictor 初始化', test_test_script_predictor_init),
    ]

    results = {}
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            results[name] = True
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    print(f"\n{'=' * 60}")
    passed = sum(results.values())
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n结果: {passed}/{len(results)} 通过")
