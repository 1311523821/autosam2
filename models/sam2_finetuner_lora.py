"""
SAM 2 LoRA 微调模型

使用 build_sam2 加载底层 Base 模型（支持梯度），而不是 Video Predictor API。
对 Image Encoder 注入 LoRA 适配器，Mask Decoder 全量微调。

训练：单帧静态微调（将视频帧拆分为单张图片）
推理：将微调后的权重用于 Video Predictor

用法:
    from models.sam2_finetuner_lora import SAM2LoRAFineTuner

    model = SAM2LoRAFineTuner(
        sam2_config='configs/sam2.1/sam2.1_hiera_t.yaml',
        sam2_checkpoint='checkpoints/sam2.1_hiera_tiny.pt',
        lora_rank=4
    )

    # 训练
    loss = model.train_step(image, point, gt_mask, optimizer, loss_fn)

    # 保存权重用于视频推理
    model.save_for_inference("finetuned_sam2.pth")
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


# ============================================================================
# LoRA Layer Implementation
# ============================================================================

class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 线性层

    在原始线性层旁边添加低秩适配器：
    output = original_linear(x) + (x @ A^T @ B^T) * scaling

    Args:
        original_linear: 原始线性层（将被冻结）
        r: LoRA 秩
        lora_alpha: LoRA 缩放因子
    """

    def __init__(self, original_linear: nn.Linear, r: int = 4, lora_alpha: int = 8):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # 低秩矩阵 A 和 B（先创建在 CPU，在 _apply 中会被移动到正确设备）
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # 初始化：A 使用 Kaiming，B 初始化为 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # 冻结原始线性层
        for param in self.original_linear.parameters():
            param.requires_grad = False

        # 将 LoRA 参数移动到与原层相同的设备
        orig_device = next(self.original_linear.parameters()).device
        self.lora_A.data = self.lora_A.data.to(orig_device)
        self.lora_B.data = self.lora_B.data.to(orig_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始分支（冻结） + LoRA 分支（可训练）
        original_out = self.original_linear(x)
        lora_out = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return original_out + lora_out

    def merge_weights(self) -> None:
        """将 LoRA 权重合并到原始线性层（用于推理）"""
        with torch.no_grad():
            self.original_linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling


def inject_lora_to_linear(
    module: nn.Module,
    target_names: List[str] = None,
    r: int = 4,
    lora_alpha: int = 8,
    verbose: bool = True
) -> int:
    """
    递归遍历模块，将匹配的 Linear 层替换为 LoRALinear

    Args:
        module: 要遍历的模块
        target_names: 要替换的层名模式列表（如 ['qkv', 'proj']）
        r: LoRA 秩
        lora_alpha: LoRA 缩放因子
        verbose: 是否打印注入信息

    Returns:
        注入的 LoRA 层数量
    """
    if target_names is None:
        target_names = ['qkv', 'proj']  # 默认注入到注意力层的 qkv 和 proj

    injected_count = 0

    def _inject_recursive(parent: nn.Module, name_prefix: str = ''):
        nonlocal injected_count

        for name, child in parent.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name

            # 检查是否是需要注入的 Linear 层
            if isinstance(child, nn.Linear):
                # 检查层名是否匹配目标模式
                should_inject = any(target in name.lower() for target in target_names)
                if should_inject:
                    lora_layer = LoRALinear(child, r=r, lora_alpha=lora_alpha)
                    setattr(parent, name, lora_layer)
                    injected_count += 1
                    if verbose:
                        print(f"  ✓ 注入 LoRA: {full_name} (r={r})")
            else:
                # 递归处理子模块
                _inject_recursive(child, full_name)

    _inject_recursive(module)
    return injected_count


# ============================================================================
# Loss Functions
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss for small target segmentation"""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)

        pred_flat = pred.flatten(1)
        target_flat = target.flatten(1)

        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)

        pos_loss = -self.alpha * (1 - pred_prob) ** self.gamma * target * F.logsigmoid(pred)
        neg_loss = -(1 - self.alpha) * pred_prob ** self.gamma * (1 - target) * F.logsigmoid(-pred)

        loss = pos_loss + neg_loss
        return loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss：Dice Loss 的泛化版本

    通过 α 和 β 控制 FP/FN 的惩罚权重：
    - α > 0.5: 更强调减少漏检（FN），适合小目标
    - β > 0.5: 更强调减少误检（FP）

    对小目标分割，推荐 α=0.7, β=0.3
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)

        pred_flat = pred.flatten(1)
        target_flat = target.flatten(1)

        tp = (pred_flat * target_flat).sum(1)
        fp = (pred_flat * (1 - target_flat)).sum(1)
        fn = ((1 - pred_flat) * target_flat).sum(1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - tversky.mean()


class CombinedLoss(nn.Module):
    """组合 Loss：支持 Dice/Tversky + Focal"""

    def __init__(self, dice_weight: float = 1.0, focal_weight: float = 1.0,
                 focal_gamma: float = 2.0, loss_type: str = 'tversky',
                 tversky_alpha: float = 0.7, tversky_beta: float = 0.3):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.loss_type = loss_type

        if loss_type == 'tversky':
            self.seg_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        else:
            self.seg_loss = DiceLoss()

        self.focal_loss = FocalLoss(gamma=focal_gamma)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        seg = self.seg_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.dice_weight * seg + self.focal_weight * focal


# ============================================================================
# SAM2 LoRA FineTuner
# ============================================================================

class SAM2LoRAFineTuner(nn.Module):
    """
    SAM 2 LoRA 微调模型

    策略：
    1. 使用 build_sam2 加载底层模型（支持梯度）
    2. 对 Image Encoder 的高层注入 LoRA（r=4）
    3. Mask Decoder 全量微调
    4. 其他模块冻结
    """

    def __init__(
        self,
        sam2_config: str,
        sam2_checkpoint: str,
        device: str = 'cuda',
        lora_rank: int = 4,
        lora_alpha: int = 8,
        lora_targets: List[str] = None,
        inject_stages: List[int] = None,  # 只在指定 stage 注入 LoRA
        finetune_memory: str = 'none'     # 'none', 'full', 'lora'
    ):
        super().__init__()
        self.device = device
        self.sam2_config = sam2_config
        self.sam2_checkpoint = sam2_checkpoint
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.finetune_memory = finetune_memory

        # 加载底层 SAM 2 模型
        self.model = self._load_model(sam2_config, sam2_checkpoint, device)

        # 注入 LoRA 到 Image Encoder
        if lora_targets is None:
            lora_targets = ['qkv', 'proj']  # 默认注入到注意力层

        if inject_stages is None:
            inject_stages = [2, 3]  # 默认只注入到 stage 2 和 stage 3（高层）

        self._inject_lora_to_encoder(lora_rank, lora_alpha, lora_targets, inject_stages)

        # 应用冻结策略
        self._apply_freeze_strategy()

        self._print_trainable_params()

    def _load_model(self, config: str, checkpoint: str, device: str):
        """加载 SAM 2 底层模型（兼容不同版本的checkpoint）"""
        try:
            from sam2.build_sam import build_sam2

            # 先构建模型（不加载checkpoint，避免版本不匹配）
            try:
                model = build_sam2(config, checkpoint, device=device)
            except RuntimeError as e:
                if 'Unexpected key' in str(e) or 'Missing key' in str(e):
                    print(f"  检测到checkpoint版本不匹配，使用宽松模式加载...")
                    # 构建模型，手动加载
                    model = build_sam2(config, device=device)
                    ckpt = torch.load(checkpoint, map_location=device)
                    if 'model' in ckpt:
                        sd = ckpt['model']
                    elif 'model_state_dict' in ckpt:
                        sd = ckpt['model_state_dict']
                    else:
                        sd = ckpt
                    model.load_state_dict(sd, strict=False)
                else:
                    raise

            print(f"SAM 2 模型加载成功: {checkpoint}")
            return model
        except Exception as e:
            raise RuntimeError(f"SAM 2 加载错误: {e}")

    def _inject_lora_to_encoder(
        self,
        lora_rank: int,
        lora_alpha: int,
        lora_targets: List[str],
        inject_stages: List[int]
    ):
        """对 Image Encoder 注入 LoRA"""
        print(f"\n注入 LoRA (r={lora_rank}, alpha={lora_alpha})...")

        injected_total = 0

        if hasattr(self.model, 'image_encoder') and hasattr(self.model.image_encoder, 'trunk'):
            trunk = self.model.image_encoder.trunk

            if hasattr(trunk, 'blocks'):
                blocks = trunk.blocks
                total_blocks = len(blocks)

                # 从 Hiera 配置动态获取 stage boundaries
                # stages: [1, 2, 7, 2] → stage_ends = [1, 3, 10, 12]
                if hasattr(trunk, 'stage_ends'):
                    stage_ends = list(trunk.stage_ends)
                else:
                    # 回退：从 stages 属性推导
                    stage_ends = [1, 3, 10, 12]

                # 转换为起始索引
                stage_boundaries = [0] + stage_ends

                for stage_idx in inject_stages:
                    if stage_idx >= len(stage_boundaries) - 1:
                        continue

                    start_block = stage_boundaries[stage_idx]
                    end_block = min(stage_boundaries[stage_idx + 1], total_blocks)

                    stage_count = 0
                    for block_idx in range(start_block, end_block):
                        block = blocks[block_idx]
                        count = inject_lora_to_linear(
                            block,
                            target_names=lora_targets,
                            r=lora_rank,
                            lora_alpha=lora_alpha,
                            verbose=False
                        )
                        stage_count += count

                    injected_total += stage_count
                    print(f"  Stage {stage_idx} (blocks {start_block}-{end_block-1}): 注入 {stage_count} 个 LoRA 层")

        print(f"总计注入 {injected_total} 个 LoRA 层")

    def _apply_freeze_strategy(self):
        """应用冻结策略"""
        print("\n应用冻结策略...")

        # 1. 先冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. 解冻 LoRA 参数（仅 lora_A 和 lora_B）
        lora_param_count = 0
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                for param in module.original_linear.parameters():
                    param.requires_grad = False
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True
                lora_param_count += module.lora_A.numel() + module.lora_B.numel()

        print(f"✓ LoRA 参数已解冻: {lora_param_count / 1e6:.2f}M")

        # 3. 解冻 Mask Decoder 全部参数
        decoder_param_count = 0
        if hasattr(self.model, 'sam_mask_decoder'):
            for param in self.model.sam_mask_decoder.parameters():
                param.requires_grad = True
                decoder_param_count += param.numel()
            print(f"✓ Mask Decoder 已解冻: {decoder_param_count / 1e6:.2f}M")

        # 4. 可选：Memory Attention 微调
        mem_param_count = 0
        if self.finetune_memory != 'none' and hasattr(self.model, 'memory_attention'):
            if self.finetune_memory == 'lora':
                # Memory Attention 使用 q_proj/k_proj/v_proj/out_proj，不是 qkv/proj
                n = inject_lora_to_linear(self.model.memory_attention,
                                         ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
                                         r=self.lora_rank, lora_alpha=self.lora_alpha)
                print(f"  注入 {n} 个 LoRA 层到 Memory Attention")
                # 重新统计
                for m in self.model.modules():
                    if isinstance(m, LoRALinear) and any(p.requires_grad for p in m.parameters()):
                        mem_param_count += m.lora_A.numel() + m.lora_B.numel()
            else:
                for param in self.model.memory_attention.parameters():
                    param.requires_grad = True
                    mem_param_count += param.numel()
            print(f"✓ Memory Attention 已解冻: {mem_param_count / 1e6:.2f}M ({self.finetune_memory})")

    def _print_trainable_params(self):
        """打印可训练参数统计"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"\n参数统计:")
        print(f"  总参数: {total_params / 1e6:.2f}M")
        print(f"  可训练: {trainable_params / 1e6:.2f}M")
        print(f"  冻结: {frozen_params / 1e6:.2f}M")
        print(f"  训练比例: {trainable_params / total_params * 100:.2f}%")

    def get_trainable_parameters(self):
        """获取可训练参数"""
        return [p for p in self.model.parameters() if p.requires_grad]

    def forward_single_frame(
        self,
        image: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单帧前向传播（可微分）

        Args:
            image: (B, 3, H, W) 输入图像
            point_coords: (B, N, 2) Prompt 点坐标
            point_labels: (B, N) Prompt 点标签 (1=前景点)

        Returns:
            low_res_masks: (B, 1, H//4, W//4) 低分辨率 mask
            iou_predictions: (B, 1) IoU 预测
        """
        # 1. 图像编码（使用 forward_image 以正确投影高分辨率特征）
        backbone_out = self.model.forward_image(image)

        # 获取图像特征
        if isinstance(backbone_out, dict):
            # forward_image 已经应用了 conv_s0 和 conv_s1 投影
            # 取最后2层特征用于 mask decoder
            fpn_feats = backbone_out["backbone_fpn"]
            image_embed = fpn_feats[-1]  # 最后一层特征 (1/16)
            high_res_features = [fpn_feats[0], fpn_feats[1]]  # (1/4, 1/8)
        else:
            image_embed = backbone_out
            high_res_features = None

        # 2. Prompt 编码（确保 batch 维度正确：(B, N, 2) 和 (B, N)）
        if point_coords.dim() == 2:
            point_coords = point_coords.unsqueeze(0)
        if point_labels.dim() == 1:
            point_labels = point_labels.unsqueeze(0)

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None
        )

        # 3. Mask 解码
        image_pe = self.model.sam_prompt_encoder.get_dense_pe()

        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features
        )

        return low_res_masks, iou_predictions

    def forward_clip(
        self,
        images: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Clip 前向传播（4帧，使用 Memory Attention）

        利用 SAM2 内部的 Memory Attention 进行帧间特征传播。
        第0帧用作记忆，后续帧通过 Memory Attention 条件化。

        Args:
            images: (B, T, 3, H, W) 或 (T, 3, H, W) 4帧clip
            point_coords: (T, N, 2) 每帧的prompt点
            point_labels: (T, N) 每帧的prompt标签

        Returns:
            all_masks: (T, 1, H//4, W//4)
            all_iou:  (T, 1)
        """
        if images.dim() == 4:
            images = images.unsqueeze(0)  # (1, T, 3, H, W)
        T = images.shape[1]

        # 1. 每帧独立编码
        all_feats = []
        for t in range(T):
            bb = self.model.forward_image(images[:, t])
            all_feats.append(bb)

        # 2. 第0帧用 prompt 生成初始 mask 和 memory
        bb0 = all_feats[0]
        image_embed0 = bb0["backbone_fpn"][-1]
        hr0 = [bb0["backbone_fpn"][0], bb0["backbone_fpn"][1]]

        pc0 = point_coords[0:1] if point_coords.dim() == 2 else point_coords[0].unsqueeze(0)
        pl0 = point_labels[0:1] if point_labels.dim() == 1 else point_labels[0].unsqueeze(0)
        if pc0.dim() == 2:
            pc0 = pc0.unsqueeze(0)
        if pl0.dim() == 1:
            pl0 = pl0.unsqueeze(0)

        sparse_emb, dense_emb = self.model.sam_prompt_encoder(
            points=(pc0, pl0), boxes=None, masks=None
        )
        image_pe = self.model.sam_prompt_encoder.get_dense_pe()

        mask0, iou0, _, _ = self.model.sam_mask_decoder(
            image_embeddings=image_embed0, image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb, dense_prompt_embeddings=dense_emb,
            multimask_output=False, repeat_image=False, high_res_features=hr0
        )

        all_masks = [mask0]
        all_iou = [iou0]

        # 3. 构建 memory（使用第0帧的输出）
        # SAM2 内部使用 _prepare_memory_conditioned_features 处理后续帧
        if T > 1 and hasattr(self.model, 'memory_attention'):
            # 提取 vision features 用于 memory
            _, vision_feats0, vision_pos_embeds0, _ = self.model._prepare_backbone_features(bb0)

            # 后续帧通过 memory attention 条件化
            for t in range(1, T):
                bb_t = all_feats[t]
                _, vision_feats_t, vision_pos_embeds_t, _ = self.model._prepare_backbone_features(bb_t)

                # Memory Attention: 用历史特征条件化当前帧
                # pix_feat_with_mem: (HW, B, C) 包含记忆信息的特征
                pix_feat_with_mem = self.model.memory_attention(
                    curr=vision_feats_t,          # 当前帧特征
                    curr_pos=vision_pos_embeds_t,  # 当前帧位置编码
                    memory=vision_feats0,          # 历史帧记忆
                    memory_pos=vision_pos_embeds0, # 历史帧位置编码
                    num_frames=T
                )

                # 将条件化后的特征重新整形用于 mask decoder
                # 注意：这一步较为 hacky，因为 SAM2 内部的特征处理比较隐式
                # 直接使用原始 backbone 特征 decode
                image_embed_t = bb_t["backbone_fpn"][-1]
                hr_t = [bb_t["backbone_fpn"][0], bb_t["backbone_fpn"][1]]

                # 获取当前帧的 prompt
                pc_t = point_coords[t].unsqueeze(0) if point_coords.dim() == 2 else point_coords[t:t+1]
                pl_t = point_labels[t].unsqueeze(0) if point_labels.dim() == 1 else point_labels[t:t+1]
                if pc_t.dim() == 2:
                    pc_t = pc_t.unsqueeze(0)
                if pl_t.dim() == 1:
                    pl_t = pl_t.unsqueeze(0)

                se, de = self.model.sam_prompt_encoder(
                    points=(pc_t, pl_t), boxes=None, masks=None
                )

                mask_t, iou_t, _, _ = self.model.sam_mask_decoder(
                    image_embeddings=image_embed_t, image_pe=image_pe,
                    sparse_prompt_embeddings=se, dense_prompt_embeddings=de,
                    multimask_output=False, repeat_image=False, high_res_features=hr_t
                )

                all_masks.append(mask_t)
                all_iou.append(iou_t)

                # 更新 memory
                vision_feats0 = vision_feats_t
                vision_pos_embeds0 = vision_pos_embeds_t

        # 堆叠结果
        masks_out = torch.cat(all_masks, dim=0)  # (T, 1, h, w)
        iou_out = torch.cat(all_iou, dim=0)      # (T, 1)
        return masks_out, iou_out

    def train_step(
        self,
        image: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        gt_mask: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        grad_clip: float = 0.0,
        scaler: object = None,
        second_optimizer: object = None,
        skip_step: bool = False,
        clip_mode: bool = False  # True: 使用 forward_clip (Memory Attention)
    ) -> Dict[str, float]:
        """
        训练步骤（支持 AMP + 双优化器 + 梯度累积 + clip模式）

        Args:
            image: (B, 3, H, W) 或 clip模式 (T, 3, H, W)
            point_coords: 单帧 (B,N,2) 或 clip (T,N,2)
            point_labels: 单帧 (B,N) 或 clip (T,N)
            gt_mask: 单帧 (B,1,H,W) 或 clip (T,1,H,W)
            optimizer: 优化器
            loss_fn: 损失函数
            grad_clip: 梯度裁剪阈值
            scaler: AMP GradScaler
            second_optimizer: 双优化器时的第二个
            skip_step: True时只累积梯度不更新参数
            clip_mode: True时使用 forward_clip

        Returns:
            {'loss': ..., 'valid': True}
        """
        if second_optimizer is None and not skip_step:
            optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            if clip_mode:
                low_res_masks, iou_pred = self.forward_clip(image, point_coords, point_labels)
            else:
                low_res_masks, iou_pred = self.forward_single_frame(image, point_coords, point_labels)

            if low_res_masks.shape[-2:] != gt_mask.shape[-2:]:
                pred_masks = F.interpolate(
                    low_res_masks, size=gt_mask.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            else:
                pred_masks = low_res_masks

            loss = loss_fn(pred_masks, gt_mask)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if skip_step:
            return {'loss': loss.item(), 'valid': True}

        if grad_clip > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                if second_optimizer is not None:
                    scaler.unscale_(second_optimizer)
            torch.nn.utils.clip_grad_norm_(self.get_trainable_parameters(), grad_clip)

        if scaler is not None:
            scaler.step(optimizer)
            if second_optimizer is not None:
                scaler.step(second_optimizer)
            scaler.update()
        else:
            optimizer.step()
            if second_optimizer is not None:
                second_optimizer.step()

        return {'loss': loss.item(), 'valid': True}

    def save_for_inference(self, save_path: str, merge_lora: bool = True):
        """
        保存权重用于推理（不修改内存中的模型）

        构建一个干净的 state_dict，将 LoRA 权重合并到原始层后，
        key 替换为原始 Linear 格式。内存中的模型保持不变。

        Args:
            save_path: 保存路径
            merge_lora: 是否将 LoRA 权重合并到原始层
        """
        if not merge_lora:
            torch.save({'model': self.model.state_dict(), 'lora_rank': self.lora_rank}, save_path)
            print(f"模型已保存到: {save_path}")
            return

        # 构建合并后的干净 state_dict（不修改模型参数）
        clean_state_dict = {}
        lora_mapping = {}  # 记录 LoRALinear 模块路径

        # 收集所有 LoRALinear 模块路径
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear):
                lora_mapping[name] = module

        for key, value in self.model.state_dict().items():
            if '.lora_A' in key or '.lora_B' in key:
                # 跳过 LoRA 参数
                continue

            if '.original_linear.' in key:
                # 将 original_linear.weight/bias 的 key 替换为标准 Linear 格式
                # 例如: blocks.3.attn.qkv.original_linear.weight -> blocks.3.attn.qkv.weight
                new_key = key.replace('.original_linear', '')
                param_name = key.split('.original_linear.')[-1]  # weight or bias

                # 提取模块路径
                module_path = key.rsplit('.original_linear.', 1)[0]

                if module_path in lora_mapping and param_name == 'weight':
                    lora_module = lora_mapping[module_path]
                    # 计算合并后的权重
                    with torch.no_grad():
                        delta = (lora_module.lora_B.data @ lora_module.lora_A.data) * lora_module.scaling
                        merged = value.clone() + delta
                    clean_state_dict[new_key] = merged
                else:
                    clean_state_dict[new_key] = value.clone()
            else:
                clean_state_dict[key] = value.clone()

        torch.save({'model': clean_state_dict, 'lora_rank': self.lora_rank}, save_path)
        print(f"模型已保存到: {save_path} (LoRA已合并到推理权重中)")


def build_sam2_lora_finetuner(
    sam2_config: str = 'configs/sam2.1/sam2.1_hiera_t.yaml',
    sam2_checkpoint: str = 'checkpoints/sam2.1_hiera_tiny.pt',
    device: str = 'cuda',
    lora_rank: int = 4
) -> SAM2LoRAFineTuner:
    """构建 SAM2 LoRA 微调模型"""
    return SAM2LoRAFineTuner(
        sam2_config=sam2_config,
        sam2_checkpoint=sam2_checkpoint,
        device=device,
        lora_rank=lora_rank
    )


if __name__ == '__main__':
    # 测试模型构建
    print("=" * 60)
    print("SAM2 LoRA 微调模型测试")
    print("=" * 60)

    model = build_sam2_lora_finetuner(
        sam2_config='configs/sam2.1/sam2.1_hiera_t.yaml',
        sam2_checkpoint='/root/autosam2/checkpoints/sam2.1_hiera_tiny.pt',
        lora_rank=4
    )

    # 测试单帧训练
    print("\n测试单帧训练步骤...")
    image = torch.randn(1, 3, 1024, 1024).cuda()
    point_coords = torch.tensor([[[512, 512]]]).float().cuda()
    point_labels = torch.tensor([[1]]).int().cuda()
    gt_mask = torch.zeros(1, 1, 1024, 1024).cuda()
    gt_mask[0, 0, 480:540, 480:540] = 1

    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-5)
    loss_fn = CombinedLoss()

    result = model.train_step(image, point_coords, point_labels, gt_mask, optimizer, loss_fn)
    print(f"\n训练结果: loss={result['loss']:.4f}")

    # 检查梯度
    print("\n梯度检查:")
    grad_found = False
    for name, param in model.model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_found = True
            print(f"  {name[:50]}: grad_norm={param.grad.norm().item():.6f}")
            break

    if not grad_found:
        print("  ⚠️ 警告: 没有检测到梯度!")

    print("\n✓ 测试完成!")
