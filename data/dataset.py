"""
Dataset for Auto-Prompt SAM 2
"""

import os
import json
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict

import sys
sys.path.insert(0, '/root/e2e')

from utils.heatmap import generate_gaussian_heatmap, get_mask_center


class VideoWeakTargetDataset(Dataset):
    """
    Video dataset for small target detection.

    Each sample returns:
    - frames: (1, D, H, W) grayscale frames normalized to [0, 1]
    - masks: (1, D, H, W) binary ground truth masks
    - centers: (D, 2) center points for each frame
    - heatmap: (D, H, W) Gaussian heatmap for the center frame
    """

    def __init__(
        self,
        txt_path: str,
        root_dir: str = '',
        clip_len: int = 4,
        target_label: str = 'uuv',
        is_train: bool = True,
        heatmap_sigma: float = 10.0
    ):
        self.clip_len = clip_len
        self.is_train = is_train
        self.target_label = target_label
        self.heatmap_sigma = heatmap_sigma
        self.clips = []

        print(f"Loading dataset from {txt_path}...")
        with open(txt_path, 'r', encoding='utf-8') as f:
            video_folders = [
                os.path.join(root_dir, line.strip().replace('\\', '/'))
                for line in f.readlines()
                if line.strip()
            ]

        stride = self.clip_len

        for folder_path in video_folders:
            if not os.path.isdir(folder_path):
                continue
            valid_files = sorted([
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            if len(valid_files) < self.clip_len:
                continue
            for i in range(0, len(valid_files) - self.clip_len + 1, stride):
                frame_names = valid_files[i: i + self.clip_len]
                self.clips.append((folder_path, frame_names))

        print(f"{'[Train]' if is_train else '[Test]'} {len(self.clips)} clips, stride={stride}")

    def __len__(self):
        return len(self.clips)

    def _get_gt_mask(self, folder_path: str, img_name: str, img_size: Tuple[int, int]) -> np.ndarray:
        """Load ground truth mask from JSON annotation."""
        json_name = os.path.splitext(img_name)[0] + '.json'
        json_path = os.path.join(folder_path, json_name)
        orig_h, orig_w = img_size
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for s in data.get('shapes', []):
                    if s['label'] == self.target_label:
                        points = np.array(s['points'], dtype=np.float32)
                        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
            except Exception:
                pass
        return mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        folder_path, frame_names = self.clips[idx]
        frames, masks, centers = [], [], []

        for name in frame_names:
            # Load image
            img = cv2.imread(os.path.join(folder_path, name), cv2.IMREAD_GRAYSCALE)
            orig_h, orig_w = img.shape[:2]

            # Padding to be divisible by 32
            pad_h = (32 - orig_h % 32) % 32
            pad_w = (32 - orig_w % 32) % 32
            if pad_h > 0 or pad_w > 0:
                img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

            frames.append(img.astype(np.float32) / 255.0)

            # Load mask
            mask = self._get_gt_mask(folder_path, name, (orig_h, orig_w))
            if pad_h > 0 or pad_w > 0:
                mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            masks.append(mask.astype(np.float32))

            # Get center point
            center = get_mask_center(torch.from_numpy(mask))
            if center is not None:
                centers.append(center)
            else:
                centers.append((0, 0))

        frames_np = np.stack(frames, axis=0)
        masks_np = np.stack(masks, axis=0)

        # Data augmentation (training only)
        if self.is_train:
            if random.random() > 0.5:
                frames_np = np.flip(frames_np, axis=2).copy()
                masks_np = np.flip(masks_np, axis=2).copy()
                centers = [(w - c[0], c[1]) for c, w in zip(centers, [frames_np.shape[2]] * len(centers))]
            if random.random() > 0.5:
                frames_np = np.flip(frames_np, axis=1).copy()
                masks_np = np.flip(masks_np, axis=1).copy()
                centers = [(c[0], h - c[1]) for c, h in zip(centers, [frames_np.shape[1]] * len(centers))]

        # Generate heatmap for the center frame (frame index clip_len // 2)
        center_frame_idx = self.clip_len // 2
        H, W = frames_np.shape[1], frames_np.shape[2]
        heatmap = generate_gaussian_heatmap(
            centers[center_frame_idx], H, W, self.heatmap_sigma
        )

        centers_np = np.array(centers, dtype=np.float32)

        return (
            torch.from_numpy(frames_np).unsqueeze(0),  # (1, D, H, W)
            torch.from_numpy(masks_np).unsqueeze(0),   # (1, D, H, W)
            torch.from_numpy(centers_np),               # (D, 2)
            heatmap                                     # (H, W)
        )


class SAM2VideoDataset(Dataset):
    """
    Dataset for SAM 2 video inference.

    Loads entire videos for SAM 2 video predictor.
    """

    def __init__(
        self,
        txt_path: str,
        root_dir: str = '',
        target_label: str = 'uuv',
        max_frames: int = 100
    ):
        self.target_label = target_label
        self.max_frames = max_frames
        self.videos = []

        with open(txt_path, 'r', encoding='utf-8') as f:
            video_folders = [
                os.path.join(root_dir, line.strip().replace('\\', '/'))
                for line in f.readlines()
                if line.strip()
            ]

        for folder_path in video_folders:
            if not os.path.isdir(folder_path):
                continue
            valid_files = sorted([
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            if len(valid_files) > 0:
                self.videos.append((folder_path, valid_files[:max_frames]))

        print(f"Loaded {len(self.videos)} videos")

    def __len__(self):
        return len(self.videos)

    def _get_gt_mask(self, folder_path: str, img_name: str, img_size: Tuple[int, int]) -> np.ndarray:
        json_name = os.path.splitext(img_name)[0] + '.json'
        json_path = os.path.join(folder_path, json_name)
        orig_h, orig_w = img_size
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for s in data.get('shapes', []):
                    if s['label'] == self.target_label:
                        points = np.array(s['points'], dtype=np.float32)
                        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
            except Exception:
                pass
        return mask

    def __getitem__(self, idx: int) -> Dict:
        folder_path, frame_names = self.videos[idx]
        frames, masks, centers = [], [], []

        for name in frame_names:
            img = cv2.imread(os.path.join(folder_path, name), cv2.IMREAD_GRAYSCALE)
            orig_h, orig_w = img.shape[:2]

            # Convert to RGB for SAM 2
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            frames.append(img_rgb)

            mask = self._get_gt_mask(folder_path, name, (orig_h, orig_w))
            masks.append(mask)

            center = get_mask_center(torch.from_numpy(mask))
            centers.append(center if center is not None else (0, 0))

        return {
            'frames': np.stack(frames, axis=0),          # (T, H, W, 3)
            'masks': np.stack(masks, axis=0),            # (T, H, W)
            'centers': np.array(centers),                 # (T, 2)
            'folder': folder_path,
            'frame_names': frame_names
        }


def custom_collate_fn(batch):
    """Collate function with dynamic padding."""
    frames, masks, centers, heatmaps = zip(*batch)
    max_h = max(f.shape[2] for f in frames)
    max_w = max(f.shape[3] for f in frames)

    padded_frames, padded_masks, padded_heatmaps = [], [], []
    for f, m, h in zip(frames, masks, heatmaps):
        pad_h = max_h - f.shape[2]
        pad_w = max_w - f.shape[3]
        f_padded = np.pad(f, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode='constant')
        m_padded = np.pad(m, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode='constant')
        h_padded = np.pad(h, ((0, pad_h), (0, pad_w)), mode='constant')
        padded_frames.append(f_padded)
        padded_masks.append(m_padded)
        padded_heatmaps.append(h_padded)

    return (
        torch.from_numpy(np.stack(padded_frames)),
        torch.from_numpy(np.stack(padded_masks)),
        torch.stack([torch.from_numpy(c) for c in centers]),
        torch.stack([torch.from_numpy(h) for h in padded_heatmaps])
    )


def get_dataloader(
    txt_path: str,
    root_dir: str = '',
    clip_len: int = 4,
    target_label: str = 'uuv',
    is_train: bool = True,
    batch_size: int = 8,
    num_workers: int = 4,
    heatmap_sigma: float = 5.0
) -> DataLoader:
    """Create dataloader for training/evaluation."""
    dataset = VideoWeakTargetDataset(
        txt_path=txt_path,
        root_dir=root_dir,
        clip_len=clip_len,
        target_label=target_label,
        is_train=is_train,
        heatmap_sigma=heatmap_sigma
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )


class VideoHeatmapDataset(Dataset):
    """
    专门用于引导头训练的数据集

    与VideoWeakTargetDataset的区别：
    1. 热力图使用最后一帧（而非中间帧）
    2. 返回格式更适合热力图回归训练

    每个样本返回：
    - frames: (1, D, H, W) 归一化灰度帧
    - heatmap: (H, W) 最后一帧的高斯热力图标签
    - mask: (H, W) 最后一帧的GT掩码（用于评估）
    - center: (2,) 目标中心点坐标
    """

    def __init__(
        self,
        txt_path: str,
        root_dir: str = '',
        clip_len: int = 4,
        target_label: str = 'uuv',
        is_train: bool = True,
        heatmap_sigma: float = 3.0,  # 默认3.0，适合小目标
        img_size: Optional[Tuple[int, int]] = None  # (H, W) 可选的目标尺寸
    ):
        """
        初始化数据集

        参数:
            txt_path: 视频列表文件路径
            root_dir: 数据根目录
            clip_len: 时序帧数（默认4）
            target_label: 目标标签
            is_train: 是否为训练模式
            heatmap_sigma: 高斯热力图标准差
            img_size: 可选的目标尺寸 (H, W)，None则使用原始尺寸
        """
        self.clip_len = clip_len
        self.is_train = is_train
        self.target_label = target_label
        self.heatmap_sigma = heatmap_sigma
        self.img_size = img_size
        self.clips = []
        self._mask_cache = {}  # JSON缓存

        print(f"加载引导头训练数据集: {txt_path}")
        with open(txt_path, 'r', encoding='utf-8') as f:
            video_folders = [
                os.path.join(root_dir, line.strip().replace('\\', '/'))
                for line in f.readlines()
                if line.strip()
            ]

        # 步长=clip_len，无重叠
        stride = self.clip_len

        for folder_path in video_folders:
            if not os.path.isdir(folder_path):
                continue
            valid_files = sorted([
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            if len(valid_files) < self.clip_len:
                continue
            for i in range(0, len(valid_files) - self.clip_len + 1, stride):
                frame_names = valid_files[i: i + self.clip_len]
                self.clips.append((folder_path, frame_names))

        print(f"{'[训练]' if is_train else '[验证]'} {len(self.clips)} 个clip, stride={stride}")

        # 预加载所有JSON标注到缓存
        print("预加载JSON标注...")
        for folder_path, frame_names in self.clips:
            # 只缓存最后一帧的JSON（用于热力图）
            name = frame_names[-1]
            json_name = os.path.splitext(name)[0] + '.json'
            cache_key = (folder_path, json_name)
            if cache_key not in self._mask_cache:
                json_path = os.path.join(folder_path, json_name)
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            self._mask_cache[cache_key] = json.load(f)
                    except:
                        self._mask_cache[cache_key] = None
                else:
                    self._mask_cache[cache_key] = None
        print(f"已缓存 {len(self._mask_cache)} 个JSON文件")

    def __len__(self):
        return len(self.clips)

    def _get_gt_mask(self, folder_path: str, img_name: str, img_size: Tuple[int, int]) -> np.ndarray:
        """从缓存的JSON标注加载GT掩码"""
        json_name = os.path.splitext(img_name)[0] + '.json'
        cache_key = (folder_path, json_name)
        orig_h, orig_w = img_size
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        data = self._mask_cache.get(cache_key)
        if data is not None:
            for s in data.get('shapes', []):
                if s['label'] == self.target_label:
                    points = np.array(s['points'], dtype=np.float32)
                    cv2.fillPoly(mask, [points.astype(np.int32)], 1)
        return mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        返回:
            frames: (1, D, H, W) 帧序列
            heatmap: (H, W) 热力图标签
            mask: (H, W) GT掩码
            center: (2,) 目标中心坐标 (x, y)
        """
        folder_path, frame_names = self.clips[idx]
        frames = []

        for name in frame_names:
            # 读取灰度图
            img = cv2.imread(os.path.join(folder_path, name), cv2.IMREAD_GRAYSCALE)
            orig_h, orig_w = img.shape[:2]

            # 如果指定了目标尺寸，进行resize
            if self.img_size is not None:
                target_h, target_w = self.img_size
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            else:
                # 32对齐padding
                pad_h = (32 - orig_h % 32) % 32
                pad_w = (32 - orig_w % 32) % 32
                if pad_h > 0 or pad_w > 0:
                    img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

            frames.append(img.astype(np.float32) / 255.0)

        # 读取最后一帧的GT掩码
        last_frame_name = frame_names[-1]
        last_img = cv2.imread(os.path.join(folder_path, last_frame_name), cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = last_img.shape[:2]
        mask = self._get_gt_mask(folder_path, last_frame_name, (orig_h, orig_w))

        # 计算原始中心点
        orig_center = get_mask_center(torch.from_numpy(mask))
        if orig_center is None:
            orig_center = (0, 0)

        # 如果指定了目标尺寸，resize掩码并调整中心点坐标
        if self.img_size is not None:
            target_h, target_w = self.img_size
            # resize掩码
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            # 调整中心点坐标
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h
            center = (orig_center[0] * scale_x, orig_center[1] * scale_y)
        else:
            # 对mask也做padding
            pad_h = (32 - orig_h % 32) % 32
            pad_w = (32 - orig_w % 32) % 32
            if pad_h > 0 or pad_w > 0:
                mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            center = orig_center

        frames_np = np.stack(frames, axis=0)  # (D, H, W)

        # 数据增强（仅训练时）
        if self.is_train:
            # 水平翻转
            if random.random() > 0.5:
                frames_np = np.flip(frames_np, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()
                center = (mask.shape[1] - center[0], center[1])
            # 垂直翻转
            if random.random() > 0.5:
                frames_np = np.flip(frames_np, axis=1).copy()
                mask = np.flip(mask, axis=0).copy()
                center = (center[0], mask.shape[0] - center[1])

        # 生成高斯热力图
        H, W = mask.shape
        heatmap = generate_gaussian_heatmap(center, H, W, self.heatmap_sigma)

        return (
            torch.from_numpy(frames_np).unsqueeze(0),  # (1, D, H, W)
            heatmap,                                    # (H, W)
            torch.from_numpy(mask.astype(np.float32)), # (H, W)
            torch.tensor(center, dtype=torch.float32)  # (2,)
        )


def heatmap_collate_fn(batch):
    """
    引导头训练专用的collate函数
    优化版：假设所有图像尺寸一致，直接stack
    """
    frames, heatmaps, masks, centers = zip(*batch)

    # 直接stack，无需动态padding（所有图像尺寸一致）
    frames_tensor = torch.stack([torch.from_numpy(f) if isinstance(f, np.ndarray) else f for f in frames])
    heatmaps_tensor = torch.stack([hm if isinstance(hm, torch.Tensor) else torch.from_numpy(hm) for hm in heatmaps])
    masks_tensor = torch.stack([m if isinstance(m, torch.Tensor) else torch.from_numpy(m) for m in masks])
    centers_tensor = torch.stack(centers)

    return frames_tensor, heatmaps_tensor, masks_tensor, centers_tensor


def get_heatmap_dataloader(
    txt_path: str,
    root_dir: str = '',
    clip_len: int = 4,
    target_label: str = 'uuv',
    is_train: bool = True,
    batch_size: int = 4,
    num_workers: int = 4,
    heatmap_sigma: float = 3.0,
    img_size: Optional[Tuple[int, int]] = None,
    seed: Optional[int] = None
) -> DataLoader:
    """
    创建引导头训练专用的DataLoader

    参数:
        txt_path: 视频列表文件
        root_dir: 数据根目录
        clip_len: 时序帧数
        target_label: 目标标签
        is_train: 是否训练模式
        batch_size: 批大小
        num_workers: 工作进程数
        heatmap_sigma: 高斯热力图标准差
        img_size: 可选的目标尺寸 (H, W)
        seed: 随机种子（用于可复现性）

    返回:
        DataLoader实例
    """
    dataset = VideoHeatmapDataset(
        txt_path=txt_path,
        root_dir=root_dir,
        clip_len=clip_len,
        target_label=target_label,
        is_train=is_train,
        heatmap_sigma=heatmap_sigma,
        img_size=img_size
    )

    # 设置随机种子相关参数
    generator = None
    worker_init_fn = None

    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

        def worker_init_fn(worker_id):
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        collate_fn=heatmap_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        generator=generator,
        worker_init_fn=worker_init_fn
    )
