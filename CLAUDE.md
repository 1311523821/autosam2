# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Context Management

- **上下文压缩**：当上下文达到 60%-75% 时，使用 `/compact` 进行压缩，保持对话流畅

## Language Requirements

- **用户交流使用中文回复**
- **代码注释使用中文**，遵循以下规范：
  - **每个文件顶部**：简短的模块说明（一句话概括用途）
  - **每个公开类/函数**：docstring 说明入参、返回值、核心逻辑
  - **关键决策点**：行内注释解释"为什么这样做"（而非重复代码"做了什么"）
  - **非直观操作**：维度变换、设备转移、数值缩放等处必须注释意图
  - **禁止**：无意义的占位注释（如 `# 初始化` 后面跟 `x = 0`）、大段废弃代码注释、英文和中文混用在同一句中

## 常用命令

```bash
# 阶段一：SAM2验证
python scripts/validate_sam2.py --data-root /root/DataBscan --test-list /root/e2e/test1.txt

# 阶段二：引导头训练
python scripts/train_guidance.py --config configs/default.yaml

# 阶段二：引导头可视化
python scripts/visualize_guidance.py --test-list /root/e2e/test1.txt

# 阶段三：端到端评估
python scripts/evaluate_e2e.py --test-list /root/e2e/test1.txt
```

## 项目结构

```
/root/autosam2/
├── scripts/               # 可执行脚本
│   ├── train_guidance.py          # 阶段二：引导头训练
│   ├── evaluate_guidance.py       # 引导头评估
│   ├── visualize_guidance.py      # 引导头可视化
│   ├── evaluate_e2e.py            # 阶段三：端到端评估
│   ├── validate_sam2.py           # 阶段一：SAM2 验证
│   ├── train_sam2_lora.py         # SAM2 LoRA 训练
│   └── test_sam2_finetuned.py     # 微调后 SAM2 测试
├── models/                # 模型定义
│   ├── guidance_head.py           # STSF 引导头
│   ├── guidance_head_swin.py      # STSF + Swin 3D 混合引导头
│   ├── auto_prompt_sam2.py        # AutoPromptSAM2 端到端管线
│   ├── lora.py                    # LoRA 模块（独立、可复用）
│   └── sam2_lora_trainer.py       # SAM2 LoRA 训练器
├── utils/                 # 工具模块
│   ├── heatmap.py                 # 高斯热力图、点提取
│   ├── losses.py                  # 引导头损失函数
│   ├── metrics.py                 # NUDT 指标（IoU/nIoU/Pd/Fa）
│   └── muon.py                    # Muon 优化器
├── data/                  # 数据加载
│   └── dataset.py                 # VideoHeatmapDataset 等
├── vendor/sam2/           # SAM 2 源码（本地副本）
├── tests/                 # 测试
│   ├── test_sam2_lora.py          # LoRA 训练器测试
│   └── test_clip_training.py
├── results/               # 实验结果
│   ├── phase3_lora/               # SAM2 LoRA 训练输出
│   ├── phase3_clip/               # Clip 训练历史结果
│   └── ...
├── docs/                  # 文档
├── checkpoints/           # 模型权重
└── configs/               # 配置文件
```

## 文档维护要求

**完成任务后必须更新以下文档：**

1. **development.html** (`/root/autosam2/docs/development.html`) - 更新进度、任务状态
2. **项目计划** (`/root/.claude/plans/`) - 更新计划文件中的进度
3. **实验记录** (`/root/autosam2/docs/`) - 记录实验结果

**更新时机：**
- 完成某个阶段的关键任务
- 实验结果有重要变化
- 发现新的问题或解决方案

## Project Overview

**Auto-Prompt SAM 2** - A small target video detection framework combining a lightweight Guidance Head with SAM 2 for sonar B-scan data. The system uses temporal variance features to detect moving targets and SAM 2's memory attention for tracking.

**Architecture**: Guidance Head (STSF encoder + lightweight decoder) → Heatmap → Prompt Point → SAM 2 Tracker

**Data**: Sonar B-scan grayscale images (~1000x500), JSON polygon annotations with 'uuv' label.

## 测试先行

**强制执行规则：修改任何 `models/`、`scripts/`、`utils/` 中的代码前，必须先完成以下步骤：**

1. 在 `tests/` 目录新增或更新对应测试文件
2. 测试用例必须明确回答三个问题：
   - **验证什么**？被测功能的具体行为
   - **怎么验证**？可执行的代码步骤
   - **预期结果**？通过/失败的判断标准
3. **在开始实现前运行现有测试确认基线状态**
4. **实现完成后再次运行全部测试确认无回归**

**禁止行为：先写实现代码再补测试。** 测试和实现必须在同一轮对话中完成，审查后才能声称"完成"。

测试放在 `tests/` 目录，命名格式 `test_<功能>.py`。示例：
```python
# 测试 1: 验证梯度流经 Memory Attention
# 预期: train_clip() 后 memory_attention.lora_A.grad ≠ 0
# 测试 2: 验证 checkpoint 兼容性
# 预期: build_sam2_video_predictor 能加载 best_inference.pth
```

实现前先跑测试确认当前状态，实现后再次跑测试确认改进。

## 代码审查

写完代码或测试用例后，必须使用 subagent 针对修改的部分进行严格审查，重点评估逻辑合理性、资源效率以及向后兼容性，避免引入隐性 Bug。

审查必须对照以下 **核心检查清单 (Checklist)** 进行，并在审查报告中明确答复：

### 1. 显存与计算图安全 (VRAM & Autograd Safety)

- **历史缓存无泄漏**：时序或历史特征缓存（Memory Bank、Hidden States 等）中保存的 Tensor，其生命周期是否明确？训练时必须保留在计算图中以保证梯度流通，但推理或跨 epoch 时是否及时释放以避免 OOM？
- **梯度冻结验证**：所有不应被训练的模块是否确实通过 `requires_grad=False` 被完全锁定？新增的可训练参数是否符合预期范围？
- **梯度累积正确性**：`zero_grad()` 的调用时机是否正确？必须在每个累积周期开始时调用，而非在 `optimizer.step()` 之前。不当的时机将导致梯度被清空或跨周期污染。
- **`retain_graph` 使用**：如使用 `retain_graph=True`，必须论证其必要性。每次 backward 保留完整计算图会导致显存随序列长度线性增长。优先使用累积 loss 后单次反向。

### 2. 设备与精度一致性 (Device & Dtype Consistency)

- **无硬编码设备字符串**：禁止使用 `torch.zeros(...).cuda()`、`.to("cuda")`、`torch.amp.autocast('cuda')` 等硬编码设备名。必须从输入 Tensor 或已注册 buffer 动态获取 `device` 和 `dtype`。
- **梯度流完整性**：前向传播中禁止使用 `.data` 直接读写 Tensor（初始化阶段除外）。Autograd 无法追踪 `.data` 上的操作。
- **混合精度兼容**：损失计算和数值敏感操作（softmax、sigmoid、normalization）是否在正确的精度下执行？注意 `autocast` 上下文中某些操作会自动提升精度，不应重复转换。

### 3. 数据流完整性与死代码 (Data-Flow & Dead Code)

- **中间变量消费追踪**：所有非平凡计算产生的中间特征是否确实在后续计算中被消费？是否存在计算了但从未使用的分支？
- **跨模块维度对齐**：特征拼接、多尺度融合、时序堆叠处的 Tensor 维度是否在所有分支上严格一致？尤其注意跨模块调用时的 `in_dim`/`out_dim` 约定（如 encoder 输出维度与 decoder 输入维度的匹配）。
- **API 约定遵循**：调用外部/底层模块时，是否遵循了其文档或源码中的输入约定（如某个参数应传入原始特征而非变换后的特征）？

### 4. 无缝集成兼容性 (Integration Compatibility)

- **权重 Key 命名一致**：保存的 state_dict 中 key 名称是否与目标加载方的期望完全匹配？LoRA 合并后的 checkpoint 必须可以直接被原始模型加载，无残留的包装器前缀（如 `.original.`、`.lora_`）。
- **Checkpoint 加载验证**：使用 `strict=False` 加载时必须打印 `missing_keys` 和 `unexpected_keys` 的数量，确认差异在预期范围内。静默加载会导致权重丢失而不被发现。
- **向后兼容**：新增参数、修改默认值、变更函数签名是否会影响已有的训练脚本或 checkpoint？

### 审查报告格式

审查完成后，报告需包含：
1. **Checklist 逐项答复** — 每条通过/未通过，附简要说明
2. **发现的问题** — 按严重程度排序（致命：会导致训练崩溃或静默失效 / 严重：潜在风险 / 建议：代码质量改进）
3. **总体结论** — 是否可以通过审查