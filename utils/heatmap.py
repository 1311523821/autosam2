"""
Heatmap utilities for point prompt generation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


def generate_gaussian_heatmap(
    center: Tuple[int, int],
    height: int,
    width: int,
    sigma: float = 10.0,
    normalize: bool = True
) -> torch.Tensor:
    """
    Generate a Gaussian heatmap centered at the given point.

    Args:
        center: (x, y) coordinates of the center
        height: Height of the heatmap
        width: Width of the heatmap
        sigma: Standard deviation of the Gaussian
        normalize: Whether to normalize to [0, 1]

    Returns:
        Heatmap tensor of shape (H, W)
    """
    x, y = center
    grid_x = torch.arange(width, dtype=torch.float32)
    grid_y = torch.arange(height, dtype=torch.float32)

    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')

    heatmap = torch.exp(-((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2 * sigma ** 2))

    if normalize:
        heatmap = heatmap / heatmap.max()

    return heatmap


def generate_multi_target_heatmap(
    centers: List[Tuple[int, int]],
    height: int,
    width: int,
    sigma: float = 10.0
) -> torch.Tensor:
    """
    Generate heatmap for multiple targets.

    Args:
        centers: List of (x, y) coordinates
        height: Height of the heatmap
        width: Width of the heatmap
        sigma: Standard deviation of the Gaussian

    Returns:
        Heatmap tensor of shape (H, W)
    """
    heatmap = torch.zeros(height, width, dtype=torch.float32)

    for x, y in centers:
        single_heatmap = generate_gaussian_heatmap((x, y), height, width, sigma)
        heatmap = torch.maximum(heatmap, single_heatmap)

    return heatmap


def extract_point_from_heatmap(
    heatmap: torch.Tensor,
    threshold: Optional[float] = None
) -> Tuple[int, int]:
    """
    Extract the highest confidence point from a heatmap.

    Args:
        heatmap: Heatmap tensor of shape (B, 1, H, W) or (H, W)
        threshold: Optional threshold for valid points

    Returns:
        (x, y) coordinates of the peak
    """
    if heatmap.dim() == 4:
        heatmap = heatmap.squeeze(0).squeeze(0)
    elif heatmap.dim() == 3:
        heatmap = heatmap.squeeze(0)

    if threshold is not None:
        heatmap = heatmap * (heatmap > threshold).float()
        if heatmap.max() == 0:
            return None

    flat_idx = torch.argmax(heatmap.view(-1))
    y = flat_idx // heatmap.shape[1]
    x = flat_idx % heatmap.shape[1]

    return (x.item(), y.item())


def extract_multiple_points_from_heatmap(
    heatmap: torch.Tensor,
    num_points: int = 1,
    min_distance: int = 10
) -> List[Tuple[int, int]]:
    """
    Extract multiple peaks from a heatmap with non-maximum suppression.

    Args:
        heatmap: Heatmap tensor of shape (H, W)
        num_points: Maximum number of points to extract
        min_distance: Minimum distance between points

    Returns:
        List of (x, y) coordinates
    """
    if heatmap.dim() == 4:
        heatmap = heatmap.squeeze(0).squeeze(0)
    elif heatmap.dim() == 3:
        heatmap = heatmap.squeeze(0)

    points = []
    heatmap_np = heatmap.cpu().numpy() if heatmap.is_cuda else heatmap.numpy()

    for _ in range(num_points):
        idx = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
        y, x = idx

        if heatmap_np[y, x] <= 0:
            break

        points.append((x, y))

        # Suppress region around the peak
        y_min = max(0, y - min_distance)
        y_max = min(heatmap_np.shape[0], y + min_distance + 1)
        x_min = max(0, x - min_distance)
        x_max = min(heatmap_np.shape[1], x + min_distance + 1)
        heatmap_np[y_min:y_max, x_min:x_max] = 0

    return points


def get_mask_center(mask: torch.Tensor) -> Tuple[int, int]:
    """
    Get the center of mass of a binary mask.

    Args:
        mask: Binary mask of shape (H, W) or (1, H, W)

    Returns:
        (x, y) coordinates of the center
    """
    if mask.dim() == 3:
        mask = mask.squeeze(0)

    # Find all nonzero coordinates
    coords = torch.nonzero(mask, as_tuple=False)
    if len(coords) == 0:
        return None

    # Compute center of mass
    center_y = coords[:, 0].float().mean()
    center_x = coords[:, 1].float().mean()

    return (int(center_x.item()), int(center_y.item()))


def get_mask_bbox(mask: torch.Tensor, padding: int = 0) -> Tuple[int, int, int, int]:
    """
    Get bounding box of a binary mask.

    Args:
        mask: Binary mask of shape (H, W)
        padding: Padding to add around the bbox

    Returns:
        (x_min, y_min, x_max, y_max)
    """
    if mask.dim() == 3:
        mask = mask.squeeze(0)

    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)

    if not rows.any() or not cols.any():
        return None

    y_min, y_max = torch.where(rows)[0][[0, -1]]
    x_min, x_max = torch.where(cols)[0][[0, -1]]

    # Add padding
    H, W = mask.shape
    x_min = max(0, x_min.item() - padding)
    y_min = max(0, y_min.item() - padding)
    x_max = min(W - 1, x_max.item() + padding)
    y_max = min(H - 1, y_max.item() + padding)

    return (x_min, y_min, x_max, y_max)
