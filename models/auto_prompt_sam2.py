"""
Auto-Prompt SAM 2 完整模型

将引导头与SAM 2结合，实现自动视频目标分割。
无需人工标注，自动生成Prompt并跟踪目标。

主要特性：
- 自动Prompt生成：引导头自动定位目标
- Re-prompting双重验证：面积突变 + 距离验证
- 可选LoRA微调：领域自适应
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from .guidance_head import GuidanceHead, build_guidance_head


class AutoPromptSAM2(nn.Module):
    """
    Auto-Prompt SAM 2 视频目标分割模型

    架构组成：
        1. 引导头 (Guidance Head)：定位目标，生成Prompt点
        2. SAM 2 视频预测器：跨帧跟踪目标

    工作流程：
        视频帧 → 引导头 → 热力图 → Prompt点 → SAM 2 → 分割结果

    Re-prompting双重验证：
        1. 面积突变检测：当前面积 < 上一帧面积 * 面积阈值
        2. 距离验证：新prompt点与原点距离 > 距离阈值
    """

    def __init__(
        self,
        guidance_head: GuidanceHead,   # 引导头模型
        sam2_checkpoint: str,          # SAM 2权重路径
        sam2_config: str,              # SAM 2配置文件
        freeze_sam2: bool = True,      # 是否冻结SAM 2
        area_threshold: float = 0.3,   # 面积突变阈值
        distance_threshold: int = 10,  # 距离验证阈值（像素）
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        self.area_threshold = area_threshold
        self.distance_threshold = distance_threshold

        # 移动引导头到指定设备
        self.guidance_head = guidance_head.to(device)
        self.guidance_head.eval()

        # 加载SAM 2视频预测器
        self.predictor = self._load_sam2_predictor(sam2_checkpoint, sam2_config, device)

        # 冻结SAM 2参数
        if freeze_sam2 and self.predictor is not None:
            for param in self.predictor.parameters():
                param.requires_grad = False

    def _load_sam2_predictor(self, checkpoint: str, config: str, device: str):
        """加载SAM 2视频预测器"""
        try:
            from sam2.build_sam import build_sam2_video_predictor
            predictor = build_sam2_video_predictor(config, checkpoint, device=device)
            print(f"SAM 2预测器加载成功: {checkpoint}")
            return predictor
        except ImportError as e:
            print(f"SAM 2导入错误: {e}")
            print("请确保已安装sam2: pip install sam2")
            return None
        except Exception as e:
            print(f"SAM 2加载错误: {e}")
            return None

    def forward(
        self,
        video_frames: torch.Tensor,
        return_heatmap: bool = False
    ) -> Dict[str, Any]:
        """
        处理视频片段

        Args:
            video_frames: 视频帧，形状 (B, T, C, H, W) 或 (B, 1, D, H, W)
            return_heatmap: 是否返回引导热力图

        Returns:
            包含以下内容的字典：
                - masks: 每帧的分割结果
                - prompt_points: 生成的Prompt点
                - heatmap: 引导热力图（可选）
        """
        # 处理输入格式
        if video_frames.dim() == 4:
            # 假设输入为 (B, 1, D, H, W)，转换为视频格式
            B, C, D, H, W = video_frames.shape
            # SAM 2需要 (T, H, W, C) 格式
            video_list = []
            for b in range(B):
                frames = video_frames[b, 0]  # (D, H, W)
                frames_rgb = frames.unsqueeze(-1).expand(-1, -1, -1, 3)  # (D, H, W, 3)
                video_list.append(frames_rgb)
            # 处理第一个batch元素
            video_np = video_list[0].cpu().numpy()
        else:
            B, T, C, H, W = video_frames.shape
            video_np = video_frames[0].permute(0, 2, 3, 1).cpu().numpy()

        # 初始化SAM 2状态
        inference_state = self.predictor.init_state(video_np)

        # 生成引导热力图（使用前4帧）
        guidance_clip = video_frames[:, :, :4] if video_frames.dim() == 5 else video_frames[:, :, :4]
        heatmap = self.guidance_head(guidance_clip)

        # 提取Prompt点
        prompt_point = self.guidance_head.get_prompt_point(heatmap)

        # 转换为numpy格式供SAM 2使用
        point_np = prompt_point[0].cpu().numpy().reshape(1, 2)

        # 向SAM 2注入Prompt
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=point_np,
            labels=np.array([1], dtype=np.int32)  # 1表示正样本点
        )

        # 视频传播跟踪
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = out_mask_logits[0]  # (1, H, W)

        result = {
            'masks': video_segments,
            'prompt_points': prompt_point,
        }

        if return_heatmap:
            result['heatmap'] = heatmap

        return result

    def process_video_with_reprompt(
        self,
        video_folder: str,
        guidance_frames: Optional[np.ndarray] = None
    ) -> Tuple[Dict[int, torch.Tensor], int]:
        """
        带Re-prompting双重验证的视频处理

        双重验证机制：
        1. 面积突变检测：当前面积 < 上一帧面积 * area_threshold
        2. 距离验证：新prompt点与原点距离 > distance_threshold

        Args:
            video_folder: 视频文件夹路径（SAM2直接从文件夹加载）
            guidance_frames: 可选的预加载帧 (T, H, W) 用于引导头

        Returns:
            video_segments: 帧索引到分割结果的映射
            reprompt_count: Re-prompting触发次数
        """
        # 初始化SAM 2状态（从文件夹路径加载）
        # 启用CPU卸载以节省GPU内存
        inference_state = self.predictor.init_state(
            video_folder,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True
        )
        T = inference_state['num_frames']
        orig_h = inference_state['video_height']
        orig_w = inference_state['video_width']

        # 加载灰度帧用于引导头（resize到256x256）
        if guidance_frames is None:
            guidance_frames = self._load_grayscale_frames(video_folder, target_size=(256, 256))

        # 计算坐标缩放因子
        scale_x = orig_w / 256.0
        scale_y = orig_h / 256.0

        # 初始引导
        initial_clip = self._extract_guidance_clip_from_frames(guidance_frames, 0)
        heatmap = self.guidance_head(initial_clip)
        prompt_point = self.guidance_head.get_prompt_point(heatmap)
        # 缩放到原始分辨率
        point_256 = prompt_point[0].cpu().numpy()
        point_np = np.array([point_256[0] * scale_x, point_256[1] * scale_y]).reshape(1, 2)

        # 注入初始Prompt
        self.predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=point_np,
            labels=np.array([1], dtype=np.int32)
        )

        # 带Re-prompting的传播跟踪
        video_segments = {}
        prev_mask_area = None
        last_point = point_np.flatten()  # 原始分辨率坐标
        reprompt_count = 0

        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            mask = (out_mask_logits[0] > 0).float()
            current_area = mask.sum().item()

            # Re-prompting双重验证
            if prev_mask_area is not None and prev_mask_area > 0:
                # 第一重：面积突变检测
                area_ratio = current_area / (prev_mask_area + 1e-6)
                if area_ratio < self.area_threshold:
                    # 获取新prompt点
                    clip = self._extract_guidance_clip_from_frames(guidance_frames, out_frame_idx)
                    new_heatmap = self.guidance_head(clip)
                    new_point = self.guidance_head.get_prompt_point(new_heatmap)
                    # 缩放到原始分辨率
                    new_point_256 = new_point[0].cpu().numpy()
                    new_point_np = np.array([new_point_256[0] * scale_x, new_point_256[1] * scale_y])

                    # 第二重：距离验证（在原始分辨率下计算）
                    distance = np.linalg.norm(new_point_np - last_point)
                    if distance > self.distance_threshold * max(scale_x, scale_y):  # 缩放距离阈值
                        # 双重验证通过，更新prompt
                        self.predictor.add_new_points(
                            inference_state=inference_state,
                            frame_idx=out_frame_idx,
                            obj_id=1,
                            points=new_point_np.reshape(1, 2),
                            labels=np.array([1], dtype=np.int32)
                        )
                        last_point = new_point_np
                        reprompt_count += 1

            video_segments[out_frame_idx] = mask
            prev_mask_area = current_area

        # 清理inference_state释放内存
        del inference_state
        torch.cuda.empty_cache()

        return video_segments, reprompt_count

    def _load_grayscale_frames(self, video_folder: str, target_size: tuple = (256, 256)) -> np.ndarray:
        """从文件夹加载灰度帧并resize到训练时的分辨率"""
        import cv2
        img_files = sorted([
            f for f in os.listdir(video_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        frames = []
        for img_file in img_files:
            img_path = os.path.join(video_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Resize到训练时的分辨率
                img = cv2.resize(img, (target_size[1], target_size[0]))
                frames.append(img.astype(np.float32) / 255.0)
        return np.stack(frames) if frames else np.zeros((1, 256, 256), dtype=np.float32)

    def _extract_guidance_clip_from_frames(
        self,
        frames: np.ndarray,
        start_idx: int
    ) -> torch.Tensor:
        """从灰度帧数组提取4帧clip用于引导头"""
        T, H, W = frames.shape
        end_idx = min(start_idx + 4, T)
        clip = frames[start_idx:end_idx]

        # 如果不足4帧，用最后一帧填充
        if clip.shape[0] < 4:
            pad = np.repeat(clip[-1:], 4 - clip.shape[0], axis=0)
            clip = np.concatenate([clip, pad], axis=0)

        # 转换为张量 (B, 1, D, H, W)
        clip_tensor = torch.from_numpy(clip).float().unsqueeze(0).unsqueeze(0)
        if self.device == 'cuda' and torch.cuda.is_available():
            clip_tensor = clip_tensor.cuda()

        return clip_tensor

    def _extract_guidance_clip(
        self,
        video_frames: np.ndarray,
        start_idx: int
    ) -> torch.Tensor:
        """提取4帧视频片段用于引导头"""
        T, H, W, C = video_frames.shape

        # 获取从start_idx开始的4帧
        end_idx = min(start_idx + 4, T)
        clip = video_frames[start_idx:end_idx]

        # 如果不足4帧，用最后一帧填充
        if clip.shape[0] < 4:
            pad = np.repeat(clip[-1:], 4 - clip.shape[0], axis=0)
            clip = np.concatenate([clip, pad], axis=0)

        # 转换为灰度图（如果需要）
        if C == 3:
            clip = np.mean(clip, axis=-1, keepdims=True)

        # 转换为张量
        clip_tensor = torch.from_numpy(clip).float().permute(3, 0, 1, 2).unsqueeze(0)

        if self.device == 'cuda' and torch.cuda.is_available():
            clip_tensor = clip_tensor.cuda()

        return clip_tensor


def build_auto_prompt_sam2(
    guidance_config: dict,
    guidance_checkpoint: str,         # 新增：引导头权重路径
    sam2_checkpoint: str,
    sam2_config: str,
    freeze_sam2: bool = True,
    area_threshold: float = 0.3,
    distance_threshold: int = 10,
    device: str = 'cuda'
) -> AutoPromptSAM2:
    """
    从配置构建AutoPromptSAM2模型

    Args:
        guidance_config: 引导头配置字典
        guidance_checkpoint: 引导头权重路径
        sam2_checkpoint: SAM 2权重路径
        sam2_config: SAM 2配置文件路径
        freeze_sam2: 是否冻结SAM 2权重
        area_threshold: 面积突变阈值（触发Re-prompt）
        distance_threshold: 距离验证阈值（像素）
        device: 运行设备

    Returns:
        AutoPromptSAM2模型实例
    """
    # 构建引导头
    guidance_head = build_guidance_head(guidance_config)

    # 加载引导头权重
    if guidance_checkpoint and os.path.exists(guidance_checkpoint):
        checkpoint = torch.load(guidance_checkpoint, map_location=device)
        # 处理不同格式的checkpoint
        if 'model_state_dict' in checkpoint:
            guidance_head.load_state_dict(checkpoint['model_state_dict'])
        else:
            guidance_head.load_state_dict(checkpoint)
        print(f"引导头权重加载成功: {guidance_checkpoint}")
    else:
        print(f"警告: 引导头权重文件不存在: {guidance_checkpoint}")
        print("使用随机初始化的权重！")

    return AutoPromptSAM2(
        guidance_head=guidance_head,
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
        freeze_sam2=freeze_sam2,
        area_threshold=area_threshold,
        distance_threshold=distance_threshold,
        device=device
    )


if __name__ == '__main__':
    # 测试模型构建
    guidance_config = {
        'in_chans': 1,
        'embed_dim': 64,
        'hidden_dim': 32,
        'num_frame': 4
    }

    # 使用fold_1的checkpoint作为测试
    guidance_ckpt = '/root/autosam2/checkpoints/experiments/exp6_embed64/enc(stsf)_embed(64)_hidden(32)_opt(adamw)_lr(5e-4)_loss(focal_a2)_sigma(10)_kfold(5)/fold_1/best.pth'

    model = build_auto_prompt_sam2(
        guidance_config=guidance_config,
        guidance_checkpoint=guidance_ckpt,
        sam2_checkpoint='/root/autosam2/checkpoints/sam2.1_hiera_tiny.pt',
        sam2_config='configs/sam2.1/sam2.1_hiera_t.yaml',
        area_threshold=0.3,
        distance_threshold=10,
        device='cuda'
    )

    print("模型构建成功！")
    print(f"  - 面积阈值: {model.area_threshold}")
    print(f"  - 距离阈值: {model.distance_threshold}px")
