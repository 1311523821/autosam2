"""
SAM2 Clip Trainer — 使用完整视频管线训练（含 cross_attn）

Fork 了 Sam2VideoPredictor 并移除 @torch.inference_mode()，
使 Memory Attention 在训练时完全可导。
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict

from .sam2_finetuner_lora import SAM2LoRAFineTuner, LoRALinear, CombinedLoss, inject_lora_to_linear


class SAM2ClipTrainer(SAM2LoRAFineTuner):
    """
    使用完整视频管线训练 SAM2（Memory Attention + cross_attn 完全可导）

    init_state_from_frames → add_new_points → propagate_in_video
    所有步骤都可导，梯度流经整个模型。
    """

    def __init__(
        self,
        sam2_config: str,
        sam2_checkpoint: str,
        device: str = 'cuda',
        lora_rank: int = 4,
        lora_alpha: int = 8,
        lora_targets: List[str] = None,
        inject_stages: List[int] = None,
        finetune_memory: str = 'lora',
    ):
        # 先调用父类做 LoRA 注入
        super().__init__(
            sam2_config=sam2_config,
            sam2_checkpoint=sam2_checkpoint,
            device=device,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_targets=lora_targets,
            inject_stages=inject_stages,
            finetune_memory=finetune_memory,
        )

        # 用 LoRA 注入后的 model 创建 forked predictor
        from .sam2_video_predictor_train import SAM2VideoPredictor
        self.predictor = SAM2VideoPredictor.from_model(self.model, device=self.device)
        print("✓ Clip Trainer 初始化完成（完整 Memory Attention 可导）")

    def train_clip(
        self,
        images: torch.Tensor,        # (T, H, W) numpy frames
        point_coords: torch.Tensor,   # (T, 2) GT 中心点
        gt_masks: torch.Tensor,       # (T, H, W) GT masks
        optimizer,
        loss_fn: nn.Module,
        grad_clip: float = 0.0,
        scaler=None,
    ) -> Dict[str, float]:
        """
        训练一个 4 帧 clip：完整的视频管线训练

        Args:
            images: (T, H, W) 或 (T, 3, H, W) numpy 帧
            point_coords: (T, 2) GT 中心点
            gt_masks: (T, H, W) GT masks (原始分辨率)
            optimizer: 优化器
            loss_fn: 损失函数
            grad_clip: 梯度裁剪阈值
            scaler: AMP GradScaler

        Returns:
            {'loss': ..., 'ious': [...]}
        """
        T = len(images)
        optimizer.zero_grad()

        # 1. 初始化状态（直接传 GPU tensor，零拷贝）
        inference_state = self.predictor.init_state_from_frames(
            images,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )

        # 2. 注入初始 prompt（第一帧的 GT 中心点）
        pt0 = point_coords[0].cpu().numpy().reshape(1, 2)
        self.predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=pt0,
            labels=np.array([1], dtype=np.int32),
        )

        # 3. 视频传播（完全可导）
        total_loss = 0.0
        frame_count = 0
        results = {}

        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            mask_logits = out_mask_logits[0]  # (1, H, W)
            gt_mask = gt_masks[out_frame_idx]

            if gt_mask is None or gt_mask.sum() == 0:
                continue

            gt_tensor = torch.from_numpy(gt_mask).float().to(mask_logits.device)
            if gt_tensor.dim() == 2:
                gt_tensor = gt_tensor.unsqueeze(0).unsqueeze(0)

            if mask_logits.shape != gt_tensor.shape:
                mask_resized = F.interpolate(
                    mask_logits.unsqueeze(0),
                    size=gt_tensor.shape[-2:],
                    mode='bilinear', align_corners=False
                ).squeeze(0)
                loss = loss_fn(mask_resized, gt_tensor)
            else:
                loss = loss_fn(mask_logits, gt_tensor)

            total_loss += loss.item()
            frame_count += 1
            loss.backward(retain_graph=True)  # 保留计算图用于后续帧

        # 4. 梯度裁剪 + 更新
        if frame_count > 0 and grad_clip > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.get_trainable_parameters(), grad_clip)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        del inference_state
        torch.cuda.empty_cache()

        return {
            'loss': total_loss / max(frame_count, 1),
            'valid': frame_count > 0,
        }
