# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Context Management

- **上下文压缩**：当上下文达到 60%-75% 时，使用 `/compact` 进行压缩，保持对话流畅

## Language Requirements

- **用户交流使用中文回复**
- **代码注释使用中文**

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
├── scripts/           # 所有可执行脚本
│   ├── train_guidance.py      # 阶段二训练
│   ├── evaluate_guidance.py   # 引导头评估
│   ├── visualize_guidance.py  # 引导头可视化
│   ├── evaluate_e2e.py        # 端到端评估
│   └── validate_sam2.py       # 阶段一验证
├── models/            # 模型定义
│   ├── guidance_head.py
│   └── auto_prompt_sam2.py
├── results/           # 所有输出结果
│   ├── phase1_sam2/          # 阶段一结果
│   ├── phase2_guidance/      # 阶段二结果
│   │   ├── training/
│   │   ├── evaluation/
│   │   └── visualization/
│   └── phase3_e2e/           # 阶段三结果
├── checkpoints/       # 模型权重
└── configs/           # 配置文件
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

## 代码审查
写完代码需要使用subagent进行审查，保证逻辑的合理性，避免引入新的问题。