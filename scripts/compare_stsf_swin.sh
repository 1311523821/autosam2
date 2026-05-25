#!/bin/bash
# ============================================================
# STSF vs STSF+Swin 快速对比实验
# 使用256x256分辨率，10个epoch快速验证
# ============================================================

set -e

export PYTHONPATH="/root/autosam2:/root/e2e:$PYTHONPATH"
DATA_ROOT="/root/DataBscan"
TRAIN_LIST="/root/e2e/train1.txt"
OUTPUT_BASE="checkpoints/ablation_256"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p ${OUTPUT_BASE}

echo "============================================"
echo "STSF vs STSF+Swin 对比实验 (256x256)"
echo "============================================"

# Exp 1: 纯 STSF
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Exp 1: 纯 STSF (256x256)"
python train_guidance.py \
    --kfold 5 \
    --epochs 10 \
    --batch-size 8 \
    --embed-dim 32 \
    --hidden-dim 16 \
    --optimizer adamw \
    --heatmap-sigma 10.0 \
    --img-size 256 256 \
    --lr 5e-4 \
    --accum-steps 2 \
    --patience 10 \
    --warmup-epochs 2 \
    --weight-decay 1e-4 \
    --loss-type heatmap_focal \
    --focal-alpha 2 \
    --focal-beta 4 \
    --output ${OUTPUT_BASE}/stsf_only \
    --data-root ${DATA_ROOT} \
    --train-list ${TRAIN_LIST} \
    2>&1 | tee ${OUTPUT_BASE}/stsf_only_${TIMESTAMP}.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Exp 1 完成"

# Exp 2: STSF + Swin
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Exp 2: STSF + Swin (256x256)"
python train_guidance.py \
    --kfold 5 \
    --epochs 10 \
    --batch-size 8 \
    --embed-dim 32 \
    --hidden-dim 16 \
    --use-swin \
    --no-conv3d-mlp \
    --optimizer adamw \
    --heatmap-sigma 10.0 \
    --img-size 256 256 \
    --lr 5e-4 \
    --accum-steps 2 \
    --patience 10 \
    --warmup-epochs 2 \
    --weight-decay 1e-4 \
    --loss-type heatmap_focal \
    --focal-alpha 2 \
    --focal-beta 4 \
    --output ${OUTPUT_BASE}/stsf_swin \
    --data-root ${DATA_ROOT} \
    --train-list ${TRAIN_LIST} \
    2>&1 | tee ${OUTPUT_BASE}/stsf_swin_${TIMESTAMP}.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Exp 2 完成"

# 汇总结果
echo ""
echo "============================================"
echo "实验结果汇总"
echo "============================================"

echo "--- 纯 STSF ---"
if [ -f "${OUTPUT_BASE}/stsf_only/kfold_results.txt" ]; then
    grep -E "(平均 Hit Rate|平均 Distance|Fold [0-9])" ${OUTPUT_BASE}/stsf_only/kfold_results.txt
fi

echo ""
echo "--- STSF + Swin ---"
if [ -f "${OUTPUT_BASE}/stsf_swin/kfold_results.txt" ]; then
    grep -E "(平均 Hit Rate|平均 Distance|Fold [0-9])" ${OUTPUT_BASE}/stsf_swin/kfold_results.txt
fi

echo ""
echo "对比完成！"
