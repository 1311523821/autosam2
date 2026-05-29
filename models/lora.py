"""
LoRA (Low-Rank Adaptation) 模块

独立的、不绑定任何模型的 LoRA 实现。

设计动机：
  全量微调 SAM2 (39M~224M 参数) 显存消耗巨大且容易过拟合。
  LoRA 在冻结的 Linear 层旁添加低秩可训练矩阵 (A, B)，
  参数增量仅为原始模型的 0.1%~1%，同时保持与全量微调接近的效果。

  对于 SAM2 Memory Attention 微调场景：
  - cross_attn 的 4 层 × 4 个投影矩阵 = 16 个 Linear 层
  - 全量训练 ~0.5M 参数，LoRA (r=4) 仅需 ~0.027M 参数
  - 合并后可直接被官方 build_sam2_video_predictor 加载

数学原理：
  h = W₀x + ΔWx = W₀x + B·A·x · (α/r)
  其中 W₀ ∈ R^(d_out × d_in) 冻结，A ∈ R^(r × d_in), B ∈ R^(d_out × r) 可训练。
  r << min(d_in, d_out)，α 控制增量幅度。

模块结构：
  LoRALinear  — LoRA 线性层（包装原始 nn.Linear）
  inject_lora  — 递归遍历模型，将匹配名称的 Linear 替换为 LoRALinear
  count_lora_params — 统计 LoRA 参数分布（用于训练日志）
  get_lora_modules  — 获取所有 LoRALinear 模块列表（用于梯度检查）
"""

import math
import torch
import torch.nn as nn
from typing import List, Optional


class LoRALinear(nn.Module):
    """
    对原始 nn.Linear 添加低秩适配器的包装层。

    forward:
      output = original(x) + dropout(x) @ A^T @ B^T * (alpha / r)

    其中:
    - original(x): 冻结的原始线性变换
    - A ∈ R^(r × d_in): 降维矩阵，将输入投影到低秩空间
    - B ∈ R^(d_out × r): 升维矩阵，从低秩空间投影回输出空间
    - r << min(d_in, d_out): 秩，控制参数量
    - alpha/r: 缩放因子，用于控制 LoRA 增量的幅度

    初始化策略：
    - A 使用 Kaiming uniform 初始化（保证初始激活方差稳定）
    - B 初始化为零矩阵（使 LoRA 增量从 0 开始，不破坏预训练权重）

    合并模式：
    - 训练时：original 和 LoRA 分支独立计算，只有 LoRA 参数被优化
    - 推理时：调用 merge_weights() 将 B@A 加到 original.weight 上，
              之后 forward 等价于标准 nn.Linear，零额外开销

    为什么需要 dropout？
    - 小数据集（声纳）上 LoRA 容易过拟合，dropout 提供正则化
    """

    def __init__(
        self,
        original: nn.Linear,
        r: int = 4,
        alpha: int = 8,
        dropout: float = 0.0,
    ):
        """
        Args:
            original: 原始 nn.Linear 层（参数将被冻结）
            r: LoRA 秩，越大表达能力越强但参数越多
            alpha: 缩放因子，实际缩放 = alpha / r
            dropout: LoRA 分支的 dropout 比率（0 = 不使用）
        """
        super().__init__()
        self.original = original
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = original.in_features
        out_features = original.out_features

        # 低秩矩阵 A (r × d_in) 和 B (d_out × r)
        # 先创建在 CPU，下面用 .data 直接搬运到原始层所在设备
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # A 用 Kaiming 初始化保证合理的激活方差
        # B 用零初始化使 LoRA 分支初始输出为 0，不干扰预训练权重
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 冻结原始层 —— 只训练 LoRA 的 A 和 B
        for p in self.original.parameters():
            p.requires_grad = False

        # 将 LoRA 参数搬运到与原始层相同的设备
        # 使用 .data 而非 Parameter() 赋值，避免触发 autograd
        orig_device = next(self.original.parameters()).device
        self.lora_A.data = self.lora_A.data.to(orig_device)
        self.lora_B.data = self.lora_B.data.to(orig_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: original(x) + lora(x)

        计算路径:
          x → original(x) ─────────────────────────┐
          x → dropout(x) → @A^T → @B^T → *scale ──→ + → output

        注意: 两个分支的计算图都会被 autograd 追踪，但 optimizer 只更新
              lora_A 和 lora_B（因为 original 的 requires_grad=False）。
        """
        out = self.original(x)
        lora_out = (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return out + lora_out

    def merge_weights(self):
        """
        将 LoRA 权重"烧入"原始层（原地修改 original.weight）。

        合并后 W = W_original + B @ A * (alpha / r)

        执行时机：训练完成后，导出推理 checkpoint 之前。
        效果：合并后的模型等价于一个标准 nn.Linear，可直接被
              原始的 SAM2 predictor 加载，无需任何 LoRA 相关代码。
        注意：此操作为原地修改，不可逆。如需继续训练，不要调用此方法。
        """
        with torch.no_grad():
            self.original.weight.data += (self.lora_B @ self.lora_A) * self.scaling


def inject_lora(
    module: nn.Module,
    target_names: List[str],
    r: int = 4,
    alpha: int = 8,
    dropout: float = 0.0,
    verbose: bool = True,
) -> int:
    """
    递归遍历 module，将名称匹配 target_names 的 nn.Linear 替换为 LoRALinear。

    替换策略：
    - 只替换 nn.Linear（不处理其他层类型）
    - 匹配规则：target_names 中任意字符串出现在层名中（大小写不敏感）
    - 例如 target_names=['q_proj', 'k_proj'] 会匹配 'encoder.q_proj' 和 'decoder.k_proj'

    为什么用名称匹配而非类型匹配？
    - 同一个模型中有不同用途的 Linear 层（如 qkv 投影 vs FFN），
      通过名称可以精确定位到需要微调的注意力投影层。
    - 预设列表（LORA_TARGET_PRESETS）统一管理目标名称，避免散落各处。

    Returns:
        成功注入的 LoRA 层总数

    注意：此函数原地修改 module 的子模块结构，不创建新模型。
    """
    injected = 0
    target_lower = [t.lower() for t in target_names]

    # 使用 list() 包裹是因为迭代过程中会修改 children
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if any(t in name.lower() for t in target_lower):
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                injected += 1
                if verbose:
                    print(f"  [LoRA] {name} (r={r})")
        else:
            # 递归处理子模块（如 Sequential、ModuleList 等容器）
            injected += inject_lora(child, target_names, r=r, alpha=alpha, dropout=dropout, verbose=verbose)

    return injected


def count_lora_params(module: nn.Module) -> dict:
    """
    统计模块中的 LoRA 参数分布。

    返回的 dict 包含：
    - lora_A: A 矩阵总参数量
    - lora_B: B 矩阵总参数量
    - total_lora: LoRA 参数总量 (A + B)
    - total_trainable: 模块中所有可训练参数总量
    - total: 模块中所有参数总量（含冻结）

    用途：训练开始时打印参数统计，验证冻结策略是否生效。
    如果 total_trainable 远大于 total_lora，说明有不该训练的参数被解冻了。
    """
    stats = {"lora_A": 0, "lora_B": 0, "total_lora": 0, "total_trainable": 0, "total": 0}
    for name, param in module.named_parameters():
        stats["total"] += param.numel()
        if param.requires_grad:
            stats["total_trainable"] += param.numel()
        if "lora_A" in name:
            stats["lora_A"] += param.numel()
            stats["total_lora"] += param.numel()
        elif "lora_B" in name:
            stats["lora_B"] += param.numel()
            stats["total_lora"] += param.numel()
    return stats


def get_lora_modules(module: nn.Module) -> List[LoRALinear]:
    """
    返回模块中所有 LoRALinear 子模块。

    用途：
    - 梯度检查：遍历 LoRA 模块，验证 lora_A/lora_B 有非零梯度
    - 合并验证：确认所有 LoRA 层都在 _build_merged_state_dict 中被处理
    """
    return [m for m in module.modules() if isinstance(m, LoRALinear)]
