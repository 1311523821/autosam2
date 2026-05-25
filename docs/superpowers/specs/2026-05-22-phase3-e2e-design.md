---
name: 第三阶段端到端整合设计
description: 引导头+SAM2端到端框架，含Re-prompting双重验证机制
type: project
---

# 第三阶段：端到端整合设计

## 目标

将引导头与SAM2整合，实现端到端的小目标视频检测系统。

**输入**: 视频文件夹路径
**输出**: 每帧分割mask (SAM2输出)
**评估指标**: IoU, nIoU, Pd, Fa, FPS, 推理时间, 权重大小

---

## 任务分解

| 任务 | 文件 | 优先级 |
|------|------|--------|
| 3.1 主框架 | `models/auto_prompt_sam2.py` | P0 |
| 3.2 Re-prompting | 同上 | P0 |
| 3.3 评估脚本 | `evaluate_e2e.py` | P0 |
| 3.4 可视化 | `scripts/visualize_e2e.py` | P1 |

---

## 数据流

```
视频文件夹路径
    │
    ├──► SAM2.init_state(folder_path) → SAM2内部加载RGB帧
    │
    └──► 手动加载灰度帧 (T, H, W)
              │
              ▼
         GuidanceHead (1, 1, 4, H, W)
              │
              ▼
         Heatmap (1, 1, H, W) → argmax → (x, y)
              │
              ▼
         SAM2.add_new_points(frame_idx=0, point=(x,y))
              │
              ▼
         propagate_in_video() → {frame_idx: mask}
```

---

## 设计细节

### 1. AutoPromptSAM2 类

```python
class AutoPromptSAM2:
    def __init__(self, guidance_ckpt, sam2_ckpt, device="cuda", 
                 area_threshold=0.3, distance_threshold=10):
        # 加载引导头（冻结）
        self.guidance = GuidanceHead(
            in_chans=1, embed_dim=64, hidden_dim=32, num_frame=4
        )
        self.guidance.load_state_dict(torch.load(guidance_ckpt)['model_state_dict'])
        self.guidance.eval()
        self.area_threshold = area_threshold
        self.distance_threshold = distance_threshold

        # 加载SAM2
        self.predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            sam2_ckpt, device=device
        )

    def process_video(self, video_folder):
        """
        video_folder: 视频文件夹路径
        返回: {frame_idx: mask}
        """
        # 1. SAM2初始化状态
        inference_state = self.predictor.init_state(video_folder)
        
        # 2. 加载灰度帧给引导头
        frames = self._load_frames(video_folder)  # (T, H, W)
        
        # 3. 获取初始prompt
        clip = frames[:4].unsqueeze(0).unsqueeze(0)  # (1, 1, 4, H, W)
        heatmap = self.guidance(clip)
        point = self.guidance.get_prompt_point(heatmap)[0].cpu().numpy()
        
        # 4. 注入prompt
        self.predictor.add_new_points(
            inference_state, frame_idx=0, obj_id=1,
            points=point.reshape(1, 2), labels=np.array([1], dtype=np.int32)
        )
        
        # 5. 传播跟踪
        results = {}
        prev_area = 0
        last_point = point
        
        for frame_idx, _, mask_logits in self.predictor.propagate_in_video(inference_state):
            mask = (mask_logits[0] > 0).cpu().numpy().squeeze()
            curr_area = mask.sum()
            
            # Re-prompt检测
            if prev_area > 0 and curr_area < prev_area * self.area_threshold:
                # 获取新prompt点
                new_point = self._get_new_prompt(frames, frame_idx)
                
                # 距离验证
                if np.linalg.norm(new_point - last_point) > self.distance_threshold:
                    self.predictor.add_new_points(
                        inference_state, frame_idx=frame_idx, obj_id=1,
                        points=new_point.reshape(1, 2), labels=np.array([1])
                    )
                    last_point = new_point
            
            results[frame_idx] = mask
            prev_area = curr_area
        
        return results
```

### 2. Re-prompting 双重验证

**触发条件**：
1. 面积突变：当前面积 < 上一帧面积 * 0.3
2. 距离验证：新prompt点与原点距离 > 10像素

**面积计算**：
```python
curr_mask = (mask_logits > 0).float()
curr_area = curr_mask.sum().item()
```

### 3. 评估脚本

**输出格式**：
```json
{
    "IoU": 0.65,
    "nIoU": 0.68,
    "Pd": 0.80,
    "Fa": 0.001,
    "FPS": 10.5,
    "inference_time_ms": 95.2,
    "model_size_mb": 422.0
}
```

**指标说明**：
- `IoU`: 全局累积IoU（所有帧交集之和/并集之和）
- `nIoU`: 每图平均IoU（每帧IoU的平均值，主指标）

---

## 参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| area_threshold | 0.3 | 面积下降比例阈值 |
| distance_threshold | 10 | 点距离阈值（像素） |
| guidance_embed_dim | 64 | 引导头嵌入维度 |
| guidance_hidden_dim | 32 | 引导头隐藏层维度 |

---

## 依赖文件

- `models/guidance_head.py` - 引导头模型
- `data/dataset.py` - 数据加载
- `utils/metrics.py` - 评估指标
- `scripts/validate_sam2.py` - SAM2 API参考

---

## 验证方案

1. 先用tiny模型冻结推理测试
2. 记录指标下降幅度
3. 根据结果决定是否需要微调

```bash
python evaluate_e2e.py \
    --guidance-ckpt checkpoints/exp6_embed64/best.pth \
    --sam2-ckpt checkpoints/sam2.1_hiera_tiny.pt \
    --test-list /root/e2e/test1.txt \
    --output results/e2e_tiny.json
```
