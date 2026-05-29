"""
SAM2 LoRA 训练器

通过配置可控的 LoRA 注入目标和训练策略，直接使用 SAM2 内部模块
（绕过推理管线），确保 Memory Attention cross-attention 梯度完整流通。

核心设计：
  传统的 SAM2 微调试图通过 predictor.propagate_in_video() 进行训练，
  但推理管线会将 memory 特征 detach 后存入 inference_state 字典，
  导致 cross-attention 的 key/value 与计算图断开，梯度无法回传。

  本训练器直接调用 SAM2 的底层模块（memory_attention、memory_encoder、
  _forward_sam_heads），自行管理 memory 的构建和传递，确保全程在
  计算图中，使 cross-attention 的 LoRA 参数能接收有效梯度。

训练流水线（T 帧 clip）:
  Frame 0: image_encoder → mask_decoder → loss_0
            └→ memory_encoder(raw_backbone_feat, pred_mask) → memory_0
  Frame t: image_encoder → memory_attention(curr, memory=[mem_0..mem_{t-1}])
            └→ mask_decoder → loss_t
            └→ memory_encoder → memory_t
  反向: Σ loss_t → gradients through cross_attn → memory_encoder → image_encoder

Memory Encoder 输入约定（遵循 SAM2 源码）:
  memory_encoder(pix_feat, masks) 中的 pix_feat 必须传入**原始 backbone 特征**
  （非 memory-conditioned），这是 SAM2Base._encode_new_memory 的设计约定。
  源码位置: sam2_base.py:691 使用 current_vision_feats[-1]（未经 memory 条件化）。

用法:
    trainer = SAM2LoRATrainer(
        sam2_config='sam2_hiera_t.yaml',
        sam2_checkpoint='checkpoints/sam2.1_hiera_tiny.pt',
        lora_config={'targets': ['cross_attn'], 'r': 4, 'alpha': 8},
        train_mask_decoder=True,
        device='cuda',
    )
    result = trainer.train_step(clip_images, gt_centers, gt_masks, optimizer, loss_fn)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from contextlib import nullcontext

from .lora import LoRALinear, inject_lora, count_lora_params, get_lora_modules


# ============================================================
# 损失函数
#   针对声纳小目标（目标像素占比 < 0.1%）优化：
#   - TverskyLoss(α=0.7, β=0.3): 大幅惩罚漏检（FN），适度容忍误检（FP）
#   - FocalLoss(γ=2): 降低大量易分负样本的权重，聚焦难分样本
#   - CombinedLoss: 两者加权组合，兼顾区域重叠和像素级平衡
# ============================================================

class DiceLoss(nn.Module):
    """Dice Loss = 1 - 2|P∩G|/(|P|+|G|)，对小目标梯度稳定"""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.reshape(pred.size(0), -1)
        target_flat = target.reshape(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss: -(1-p)^γ * log(p)，降低易分样本权重，聚焦难分样本"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(pred)
        pos = -self.alpha * (1 - prob) ** self.gamma * target * F.logsigmoid(pred)
        neg = -(1 - self.alpha) * prob ** self.gamma * (1 - target) * F.logsigmoid(-pred)
        return (pos + neg).mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss — Dice 的泛化版本。

    α > 0.5 时更强调减少漏检（FN），适合小目标场景。
    默认 α=0.7, β=0.3：FN 权重是 FP 的 2.3 倍。
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.reshape(pred.size(0), -1)
        target_flat = target.reshape(target.size(0), -1)
        tp = (pred_flat * target_flat).sum(1)
        fp = (pred_flat * (1 - target_flat)).sum(1)
        fn = ((1 - pred_flat) * target_flat).sum(1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - tversky.mean()


class CombinedLoss(nn.Module):
    """Tversky/Dice + Focal 加权组合，同时优化区域重叠和像素级分类"""

    def __init__(self, seg_type: str = 'tversky', seg_weight: float = 1.0,
                 focal_weight: float = 1.0, focal_gamma: float = 2.0,
                 tversky_alpha: float = 0.7, tversky_beta: float = 0.3):
        super().__init__()
        self.seg_weight = seg_weight
        self.focal_weight = focal_weight
        if seg_type == 'tversky':
            self.seg_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        elif seg_type == 'dice':
            self.seg_loss = DiceLoss()
        else:
            raise ValueError(f"Unknown seg_type: {seg_type}")
        self.focal_loss = FocalLoss(gamma=focal_gamma)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.seg_weight > 0:
            loss += self.seg_weight * self.seg_loss(pred, target)
        if self.focal_weight > 0:
            loss += self.focal_weight * self.focal_loss(pred, target)
        return loss


# ============================================================
# LoRA 注入预设
#   每个预设定义了一组 target_modules（在哪些子模块中找）和
#   target_names（匹配哪些 Linear 层名）。通过 --lora-targets 参数
#   可以灵活组合。
# ============================================================

LORA_TARGET_PRESETS = {
    'cross_attn': {
        'description': '仅 Memory Attention 的 cross-attention 投影层',
        'target_modules': ['cross_attn_image'],
        'target_names': ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
    },
    'self_attn': {
        'description': '仅 Memory Attention 的 self-attention 投影层',
        'target_modules': ['self_attn'],
        'target_names': ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
    },
    'memory_attn': {
        'description': 'Memory Attention 全部（self + cross）',
        'target_modules': ['self_attn', 'cross_attn_image'],
        'target_names': ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
    },
    'image_encoder': {
        'description': 'Image Encoder 高层 stage',
        'target_modules': ['image_encoder.trunk.blocks'],
        'target_names': ['qkv', 'proj'],
    },
}


# ============================================================
# SAM2LoRATrainer
# ============================================================

class SAM2LoRATrainer(nn.Module):
    """
    SAM2 LoRA 训练器。

    与旧版 SAM2LoRAFineTuner / SAM2ClipTrainer 的核心区别：
    - 旧版试图通过 predictor.propagate_in_video() 训练，但 memory 被 detach
    - 新版直接调用 model.memory_attention() 等底层 API，memory 保留在计算图中
    - 不再需要 fork SAM2VideoPredictor、不再有 1287 行的 vendor 代码

    参数冻结策略（由 _apply_freeze_strategy 执行）：
    1. 冻结全部参数
    2. 解冻所有 LoRA 的 lora_A 和 lora_B
    3. 解冻 sam_mask_decoder（全量训练，因为声纳域与自然图像域差异大）
    4. 可选解冻 memory_encoder
    """

    def __init__(
        self,
        sam2_config: str,
        sam2_checkpoint: str,
        lora_config: Optional[Dict] = None,
        train_mask_decoder: bool = True,
        train_memory_encoder: bool = False,
        device: str = 'cuda',
    ):
        super().__init__()
        self.device = device
        self.sam2_config = sam2_config
        self.train_mask_decoder = train_mask_decoder

        if lora_config is None:
            lora_config = {'targets': ['cross_attn'], 'r': 4, 'alpha': 8}
        self.lora_config = lora_config

        # 加载 SAM2 底层模型（不通过 predictor API，直接拿 nn.Module）
        self.model = self._load_sam2(sam2_config, sam2_checkpoint, device)

        # 按配置注入 LoRA 适配器
        lora_targets = lora_config.get('targets', ['cross_attn'])
        lora_r = lora_config.get('r', 4)
        lora_alpha = lora_config.get('alpha', 8)
        lora_dropout = lora_config.get('dropout', 0.0)
        self._inject_lora(lora_targets, lora_r, lora_alpha, lora_dropout)

        # 应用冻结策略
        self._apply_freeze_strategy(train_memory_encoder)

        self._print_info()

    # ----------------------------------------------------------------
    # 模型加载
    # ----------------------------------------------------------------

    def _load_sam2(self, config: str, checkpoint: str, device: str):
        """
        加载 SAM2 底层模型。

        兼容不同版本的 checkpoint：如果直接加载失败（key 不匹配），
        先用配置文件构建模型，再用 strict=False 加载权重。
        """
        from sam2.build_sam import build_sam2

        try:
            model = build_sam2(config, checkpoint, device=device)
        except RuntimeError as e:
            if 'Unexpected key' in str(e) or 'Missing key' in str(e):
                print(f"  checkpoint 版本不匹配，宽松模式加载...")
                model = build_sam2(config, device=device)
                ckpt = torch.load(checkpoint, map_location=device)
                sd = ckpt.get('model', ckpt)
                model.load_state_dict(sd, strict=False)
            else:
                raise

        print(f"  SAM2 加载: {checkpoint}")
        return model

    # ----------------------------------------------------------------
    # LoRA 注入
    # ----------------------------------------------------------------

    def _inject_lora(self, targets: List[str], r: int, alpha: int, dropout: float):
        """根据预设 targets 遍历模型并注入 LoRA"""
        print(f"\n  LoRA 注入 (r={r}, alpha={alpha}):")

        total = 0
        for target in targets:
            if target not in LORA_TARGET_PRESETS:
                print(f"    未知 target: {target}，跳过")
                continue

            preset = LORA_TARGET_PRESETS[target]
            print(f"    [{target}] {preset['description']}")

            if target == 'image_encoder':
                total += self._inject_image_encoder(preset, r, alpha, dropout)
            else:
                total += self._inject_memory_attention(preset, r, alpha, dropout)

        print(f"  总计注入 {total} 个 LoRA 层")

    def _inject_memory_attention(self, preset: dict, r: int, alpha: int, dropout: float) -> int:
        """
        注入 LoRA 到 Memory Attention 的指定子模块。

        MemoryAttention 有 num_layers 个 MemoryAttentionLayer，
        每个 layer 包含 self_attn 和 cross_attn_image 两个注意力模块。
        通过 preset['target_modules'] 指定要注入到哪个注意力模块。
        """
        ma = self.model.memory_attention
        total = 0
        for i, layer in enumerate(ma.layers):
            for mod_name in preset['target_modules']:
                if hasattr(layer, mod_name):
                    mod = getattr(layer, mod_name)
                    n = inject_lora(mod, preset['target_names'], r=r, alpha=alpha,
                                    dropout=dropout, verbose=False)
                    if n > 0:
                        print(f"      layer.{i}.{mod_name}: {n} 层")
                    total += n
        return total

    def _inject_image_encoder(self, preset: dict, r: int, alpha: int, dropout: float) -> int:
        """
        注入 LoRA 到 Image Encoder 的高层 stage。

        SAM2 的 Hiera  backbone 分 4 个 stage（0-3），通过 stage_ends 划分。
        只注入 stage 2 和 stage 3（高层语义特征），
        stage 0 和 stage 1（底层纹理）保持冻结以减少参数量。
        """
        trunk = self.model.image_encoder.trunk
        total = 0

        if hasattr(trunk, 'stage_ends'):
            stage_ends = list(trunk.stage_ends)
        else:
            stage_ends = [0, 2, 9, 11]  # SAM2 Tiny 默认 stage 划分

        stage_boundaries = [0] + [se + 1 for se in stage_ends]
        for stage_idx in [2, 3]:
            if stage_idx >= len(stage_boundaries) - 1:
                continue
            start, end = stage_boundaries[stage_idx], min(stage_boundaries[stage_idx + 1], len(trunk.blocks))
            for bidx in range(start, end):
                n = inject_lora(trunk.blocks[bidx], preset['target_names'],
                                r=r, alpha=alpha, dropout=dropout, verbose=False)
                total += n
            print(f"      stage {stage_idx} (blocks {start}-{end-1}): {total} 层")

        return total

    # ----------------------------------------------------------------
    # 冻结策略
    # ----------------------------------------------------------------

    def _apply_freeze_strategy(self, train_memory_encoder: bool):
        """
        应用参数冻结策略。

        解冻顺序很重要（后面的步骤覆盖前面的）：
        1. 全部冻结 → 2. 解冻 LoRA 参数 → 3. 解冻 mask_decoder → 4. 可选解冻 memory_encoder

        memory_encoder 默认冻结，因为它的作用是压缩 mask 特征到 memory，
        声纳目标的 mask 形态与自然图像差异不大，冻结通常足够。
        """
        # 1. 冻结全部
        for p in self.model.parameters():
            p.requires_grad = False

        # 2. 解冻 LoRA 参数（A 和 B），同时确保原始权重保持冻结
        for m in self.model.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.requires_grad = True
                m.lora_B.requires_grad = True
                for p in m.original.parameters():
                    p.requires_grad = False

        # 3. 解冻 Mask Decoder（全量训练，而不只是 LoRA）
        #    声纳图像与自然图像域差异大，全量训练 mask_decoder 效果更好
        if self.train_mask_decoder:
            for p in self.model.sam_mask_decoder.parameters():
                p.requires_grad = True

        # 4. 可选：memory_encoder
        if train_memory_encoder:
            for p in self.model.memory_encoder.parameters():
                p.requires_grad = True

        # 5. no_mem_embed：第一帧使用的"空记忆"嵌入，需要跟随 domain 适配
        if hasattr(self.model, 'no_mem_embed'):
            self.model.no_mem_embed.requires_grad = True

    def _print_info(self):
        """打印可训练参数统计"""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        lora_stats = count_lora_params(self.model)

        print(f"\n  参数统计:")
        print(f"    总参数:     {total / 1e6:.2f}M")
        print(f"    可训练:     {trainable / 1e6:.2f}M ({trainable / total * 100:.1f}%)")
        print(f"    其中 LoRA:  {lora_stats['total_lora'] / 1e6:.4f}M")
        print(f"    其中 A:     {lora_stats['lora_A']}")
        print(f"    其中 B:     {lora_stats['lora_B']}")

    # ----------------------------------------------------------------
    # 可训练参数获取
    # ----------------------------------------------------------------

    def get_trainable_params(self) -> List[nn.Parameter]:
        """返回所有可训练参数，供优化器使用"""
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_param_groups(self, lr_mult: float = 10.0) -> List[Dict]:
        """
        将可训练参数分组，用于 Muon+AdamW 双优化器策略。

        Muon 适用 2D/4D 权重矩阵（通过 Newton-Schulz 正交化加速收敛），
        AdamW 适用 1D 偏置和 LayerNorm 参数。
        lora_A 是 2D（r × in_features），但形状较小，归入 AdamW 组。

        Returns:
            [{'params': [...], 'lr_scale': lr_mult},   # Muon 组（高学习率）
             {'params': [...], 'lr_scale': 1.0}]       # AdamW 组（基础学习率）
        """
        group_2d = []  # Muon
        group_1d = []  # AdamW

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # lora_A 显式归入 AdamW（小矩阵用 Muon 意义不大）
            if p.ndim in [2, 4] and 'bias' not in name and 'norm' not in name and 'lora_A' not in name:
                group_2d.append(p)
            else:
                group_1d.append(p)

        return [
            {'params': group_2d, 'lr_scale': lr_mult},
            {'params': group_1d, 'lr_scale': 1.0},
        ]

    # ----------------------------------------------------------------
    # 训练核心
    #   注意：train_step 不负责 zero_grad()。zero_grad 由外部训练循环
    #   在每个梯度累积周期开始时调用，以避免早期梯度被错误清空。
    # ----------------------------------------------------------------

    def train_step(
        self,
        clip_images: torch.Tensor,
        gt_centers: torch.Tensor,
        gt_masks: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        grad_clip: float = 1.0,
        scaler: Optional[torch.amp.GradScaler] = None,
        second_optimizer: Optional[torch.optim.Optimizer] = None,
        skip_update: bool = False,
    ) -> Dict[str, Any]:
        """
        单次训练步骤（前向 + 反向，可选参数更新）。

        Args:
            clip_images: (T, 3, H, W) 已归一化的视频帧 tensor
            gt_centers: (T, 2) 目标中心点像素坐标 (x, y)
            gt_masks: (T, H, W) 二值 GT mask
            optimizer: 主优化器
            loss_fn: 损失函数
            grad_clip: 梯度裁剪阈值（0 = 不裁剪）
            scaler: AMP GradScaler（None = 不使用 AMP）
            second_optimizer: 双优化器时的第二个（如 Muon）
            skip_update: True 时只累积梯度不调用 optimizer.step()
                        用于梯度累积——外部循环在最后一步才传 False

        Returns:
            {'loss': float, 'frame_losses': [float], 'grad_norm': float}
        """
        T = clip_images.shape[0]
        device = clip_images.device

        # 动态判断设备类型以兼容 AMP（避免硬编码 'cuda'）
        use_amp = scaler is not None
        device_type = 'cuda' if device.type == 'cuda' else device.type
        autocast_ctx = torch.amp.autocast(device_type) if use_amp else nullcontext()

        with autocast_ctx:
            loss, frame_losses = self._forward_clip(clip_images, gt_centers, gt_masks, loss_fn)

        # 反向传播（在 autocast 外部执行）
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if skip_update:
            return {'loss': loss.item(), 'frame_losses': frame_losses, 'grad_norm': 0.0}

        # 梯度裁剪
        grad_norm = 0.0
        if grad_clip > 0:
            trainable = self.get_trainable_params()
            if use_amp:
                # unscale 后才能正确计算梯度的 L2 范数
                scaler.unscale_(optimizer)
                if second_optimizer is not None:
                    scaler.unscale_(second_optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip).item()

        # 参数更新
        if use_amp:
            scaler.step(optimizer)
            if second_optimizer is not None:
                scaler.step(second_optimizer)
            scaler.update()
        else:
            optimizer.step()
            if second_optimizer is not None:
                second_optimizer.step()

        return {'loss': loss.item(), 'frame_losses': frame_losses, 'grad_norm': grad_norm}

    def _forward_clip(
        self,
        images: torch.Tensor,
        gt_centers: torch.Tensor,
        gt_masks: torch.Tensor,
        loss_fn: nn.Module,
        return_preds: bool = False,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Clip 前向传播 — 标准半监督 VOS 训练模式。

        提示策略（关键设计，与 propagate_in_video 推理对齐）：
        - 帧 0（参考帧）: GT 中心点 → 稀疏提示，匹配"用户仅标注首帧"的推理场景
        - 帧 1+（传播帧）: 不给任何人工提示（No point, No mask_input）
          mask_decoder 完全依赖 memory-conditioned features 定位目标
          cross-attention LoRA 是唯一的空间信息来源
        - 所有帧都计算 loss（与 GT mask 比较）

        梯度控制：
        - memory_encoder 的输入 mask .detach()：防止跨帧梯度链导致 OOM
        """
        T = images.shape[0]
        B = 1  # clip 模式固定 batch=1

        accumulated_memories = []  # [(features, pos_enc), ...]
        total_loss = torch.tensor(0.0, device=images.device)
        frame_losses = []
        all_preds = [] if return_preds else None
        prev_mask = None  # 上一帧预测 mask，供帧 1+ 作为 mask_inputs 密集提示

        for t in range(T):
            # --- 1. 图像编码 ---
            backbone_out = self.model.forward_image(images[t:t + 1])
            backbone_out, vis_feats, vis_pos, feat_sizes = self.model._prepare_backbone_features(
                backbone_out
            )

            # --- 2. 条件化 ---
            if t > 0 and len(accumulated_memories) > 0:
                pix_feat = self._apply_memory_attention(
                    vis_feats, vis_pos, feat_sizes, accumulated_memories
                )
            else:
                H, W = feat_sizes[-1]
                pix_feat = vis_feats[-1].permute(1, 2, 0).view(B, self.model.hidden_dim, H, W)

            # --- 3. 多尺度特征 ---
            if len(vis_feats) > 1:
                high_res_features = [
                    x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                    for x, s in zip(vis_feats[:-1], feat_sizes[:-1])
                ]
            else:
                high_res_features = None

            # --- 4. SAM head ---
            # 帧 0: GT 中心点 → 稀疏提示（匹配推理：仅首帧有标注）
            # 帧 1+: 不给任何提示，mask_decoder 纯粹从 memory-conditioned features 中
            #        定位目标——与 propagate_in_video 内部行为完全对齐
            if t == 0:
                point_inputs = {
                    "point_coords": gt_centers[0:1].unsqueeze(1).float().to(images.device),
                    "point_labels": torch.ones(B, 1, dtype=torch.int32, device=images.device),
                }
                mask_inputs_for_sam = None
            else:
                point_inputs = None
                mask_inputs_for_sam = None

            sam_outputs = self.model._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs_for_sam,
                high_res_features=high_res_features,
                multimask_output=False,
            )
            _, _, _, _, high_res_masks, _, _ = sam_outputs

            # --- 5. 损失计算（所有帧都与 GT mask 比较） ---
            gt = gt_masks[t].to(images.device)
            if gt.dim() == 2:
                gt = gt.unsqueeze(0).unsqueeze(0)
            elif gt.dim() == 3:
                gt = gt.unsqueeze(0)

            if high_res_masks.shape[-2:] != gt.shape[-2:]:
                pred_for_loss = F.interpolate(
                    high_res_masks, size=gt.shape[-2:], mode='bilinear', align_corners=False
                )
            else:
                pred_for_loss = high_res_masks

            loss_t = loss_fn(pred_for_loss, gt)
            total_loss = total_loss + loss_t
            frame_losses.append(loss_t.item())

            if return_preds:
                all_preds.append(torch.sigmoid(pred_for_loss).squeeze(0).squeeze(0))

            # --- 6. Memory Encoder ---
            # 用原始 backbone 特征 + detach 后的 mask 构建 memory
            # detach 理由：防止帧 t+1 的 loss 通过 memory → mask_t → mask_decoder_t 回传
            #            形成跨帧梯度链导致 OOM 和训练不稳定
            mask_for_mem = torch.sigmoid(high_res_masks).detach()
            prev_mask = mask_for_mem  # 保存供下一帧 mask_inputs 使用（已 detach）
            raw_pix_feat = vis_feats[-1].permute(1, 2, 0).view(B, self.model.hidden_dim, *feat_sizes[-1])
            mem_out = self.model.memory_encoder(
                pix_feat=raw_pix_feat,
                masks=mask_for_mem,
                skip_mask_sigmoid=True,
            )
            accumulated_memories.append((
                mem_out["vision_features"],
                mem_out["vision_pos_enc"][0],
            ))

        avg_loss = total_loss / T
        if return_preds:
            return avg_loss, frame_losses, all_preds
        return avg_loss, frame_losses

    def _apply_memory_attention(
        self,
        vis_feats: List[torch.Tensor],
        vis_pos: List[torch.Tensor],
        feat_sizes: List[Tuple[int, int]],
        memories: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """
        将累积的 history memory 应用到当前帧特征。

        步骤：
        1. 将所有 memory 从空间格式 (B, C, H, W) 展平为 token 序列 (N, B, C)
        2. 拼接所有 memory token
        3. 调用 self.model.memory_attention(curr, memory) 进行 self-attn + cross-attn
        4. 将输出从 token 序列 reshape 回空间格式 (B, C, H, W)

        cross-attn 维度说明（这是 SAM2 的正常设计，非 bug）：
        - query: (H*W, B, 256) — 来自 backbone 特征
        - key/value: (N_mem, B, 64) — 来自 memory_encoder 压缩后的特征
        - 不同的维度通过各自的 q_proj/k_proj/v_proj 投影到相同的内部维度

        Returns:
            条件化后的特征 (B, C, H, W)
        """
        B = vis_feats[-1].size(1)
        H, W = feat_sizes[-1]
        device = vis_feats[-1].device

        # 将所有 memory 展平并拼接
        mem_tokens = []
        mem_pos_tokens = []
        for mem_feat, mem_pos in memories:
            mem_feat = mem_feat.to(device)
            mem_pos = mem_pos.to(device)
            # (B, C, H, W) → (H*W, B, C)
            mem_tokens.append(mem_feat.flatten(2).permute(2, 0, 1))
            mem_pos_tokens.append(mem_pos.flatten(2).permute(2, 0, 1))

        all_mem = torch.cat(mem_tokens, dim=0)          # (total_tokens, B, mem_dim)
        all_mem_pos = torch.cat(mem_pos_tokens, dim=0)  # (total_tokens, B, mem_dim)

        # 只传顶层特征（最后一级）给 memory attention
        conditioned = self.model.memory_attention(
            curr=vis_feats[-1:],     # list of 1: (H*W, B, C=256)
            curr_pos=vis_pos[-1:],   # list of 1: (H*W, B, C=256)
            memory=all_mem,          # (total_tokens, B, mem_dim=64)
            memory_pos=all_mem_pos,  # (total_tokens, B, mem_dim=64)
            num_obj_ptr_tokens=0,    # 训练时不使用 object pointer
        )

        # (H*W, B, C) → (B, C, H, W)
        return conditioned.permute(1, 2, 0).view(B, self.model.hidden_dim, H, W)

    # ----------------------------------------------------------------
    # 单帧前向（验证时使用，不经过 memory attention）
    # ----------------------------------------------------------------

    def forward_single_frame(
        self,
        image: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单帧前向传播（无 memory，用于快速验证 mask_decoder 的效果）。

        Args:
            image: (B, 3, H, W)
            point_coords: (B, N, 2) 像素坐标
            point_labels: (B, N) 1=前景, 0=背景, -1=填充

        Returns:
            (low_res_masks (B, 1, 256, 256), iou_predictions (B, 1))
        """
        backbone_out = self.model.forward_image(image)
        fpn_feats = backbone_out["backbone_fpn"]
        image_embed = fpn_feats[-1]
        high_res = [fpn_feats[0], fpn_feats[1]] if len(fpn_feats) > 2 else None

        # 确保 batch 维度
        if point_coords.dim() == 2:
            point_coords = point_coords.unsqueeze(0)
        if point_labels.dim() == 1:
            point_labels = point_labels.unsqueeze(0)

        se, de = self.model.sam_prompt_encoder(
            points=(point_coords, point_labels), boxes=None, masks=None
        )
        image_pe = self.model.sam_prompt_encoder.get_dense_pe()

        masks, ious, _, _ = self.model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res,
        )
        return masks, ious

    # ----------------------------------------------------------------
    # Checkpoint 保存
    # ----------------------------------------------------------------

    def save_checkpoint(self, path: str, merge_lora: bool = True,
                        extra: Optional[Dict] = None):
        """
        保存 checkpoint。

        Args:
            path: 保存路径
            merge_lora: True=合并 LoRA 到原始权重（推理用，可直接被官方 predictor 加载）
                        False=保留 LoRA 结构（继续训练用）
            extra: 额外数据（epoch, optimizer, metrics 等）
        """
        if merge_lora:
            clean_sd = self._build_merged_state_dict()
            data = {'model': clean_sd, 'lora_config': self.lora_config}
        else:
            data = {
                'model_state_dict': self.model.state_dict(),
                'lora_config': self.lora_config,
            }

        if extra:
            data.update(extra)

        torch.save(data, path)
        tag = "推理" if merge_lora else "训练"
        print(f"  Checkpoint 已保存 ({tag}): {path}")

    def _build_merged_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        构建 LoRA 权重合并后的 state_dict。

        合并公式: W_merged = W_original + B @ A * (alpha / r)

        Key 转换规则:
        - 'xxx.lora_A' / 'xxx.lora_B' → 跳过
        - 'xxx.original.weight' → 'xxx.weight'（与原始模型 key 一致）
        - 其他 key → 保持不变
        """
        lora_modules = {}
        for name, m in self.model.named_modules():
            if isinstance(m, LoRALinear):
                lora_modules[name] = m

        clean = {}
        for key, value in self.model.state_dict().items():
            # 跳过 LoRA 参数（它们被合并到原始权重中）
            if '.lora_A' in key or '.lora_B' in key:
                continue

            if '.original.' in key:
                # 将 'xxx.original.weight' 替换为 'xxx.weight'
                new_key = key.replace('.original.', '.')
                param_type = key.rsplit('.original.', 1)[-1]  # 'weight' or 'bias'
                mod_path = key.rsplit('.original.', 1)[0]     # 模块路径

                if mod_path in lora_modules and param_type == 'weight':
                    # 合并: W = W_orig + B @ A * (alpha/r)
                    lora_m = lora_modules[mod_path]
                    delta = (lora_m.lora_B.data @ lora_m.lora_A.data) * lora_m.scaling
                    clean[new_key] = (value + delta).clone()
                else:
                    clean[new_key] = value.clone()
            else:
                clean[key] = value.clone()

        return clean


def build_trainer(
    sam2_config: str = 'sam2_hiera_t.yaml',
    sam2_checkpoint: str = 'checkpoints/sam2.1_hiera_tiny.pt',
    lora_targets: List[str] = None,
    lora_rank: int = 4,
    lora_alpha: int = 8,
    train_mask_decoder: bool = True,
    device: str = 'cuda',
) -> SAM2LoRATrainer:
    """便捷构建函数，一行创建 trainer"""
    if lora_targets is None:
        lora_targets = ['cross_attn']

    return SAM2LoRATrainer(
        sam2_config=sam2_config,
        sam2_checkpoint=sam2_checkpoint,
        lora_config={
            'targets': lora_targets,
            'r': lora_rank,
            'alpha': lora_alpha,
        },
        train_mask_decoder=train_mask_decoder,
        device=device,
    )
