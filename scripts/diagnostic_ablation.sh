#!/bin/bash
# SAM2 LoRA 微调消融实验
# 对比损失函数和优化器对微调效果的影响
set -e

BASE_DIR="/root/autosam2"
TEST_LIST="/root/e2e/test1.txt"
EPOCHS=50
VAL_EVERY=3
BATCH_SIZE=1

echo "============================================================"
echo "  SAM2 微调消融实验"
echo "  训练轮数: ${EPOCHS}, val_every: ${VAL_EVERY}"
echo "============================================================"

# ── 实验 A: Dice Loss + AdamW (基线) ──
echo ""
echo "████████████████████████████████████████████████████████████"
echo "  实验 A: Dice + AdamW (基线)"
echo "████████████████████████████████████████████████████████████"
python ${BASE_DIR}/scripts/finetune_sam2_lora.py \
    --loss-type dice \
    --optimizer adamw \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --val-every ${VAL_EVERY} \
    --patience 5 \
    --output ${BASE_DIR}/results/ablation_a

echo ""
echo "  测试 A..."
python ${BASE_DIR}/scripts/test_sam2_finetuned.py \
    --sam2-ckpt ${BASE_DIR}/results/ablation_a/checkpoints/best_inference.pth \
    --test-list ${TEST_LIST} \
    --output ${BASE_DIR}/results/ablation_a

# ── 实验 B: Tversky Loss + AdamW ──
echo ""
echo "████████████████████████████████████████████████████████████"
echo "  实验 B: Tversky + AdamW"
echo "████████████████████████████████████████████████████████████"
python ${BASE_DIR}/scripts/finetune_sam2_lora.py \
    --loss-type tversky \
    --optimizer adamw \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --val-every ${VAL_EVERY} \
    --patience 5 \
    --output ${BASE_DIR}/results/ablation_b

echo ""
echo "  测试 B..."
python ${BASE_DIR}/scripts/test_sam2_finetuned.py \
    --sam2-ckpt ${BASE_DIR}/results/ablation_b/checkpoints/best_inference.pth \
    --test-list ${TEST_LIST} \
    --output ${BASE_DIR}/results/ablation_b

# ── 实验 C: Dice Loss + Muon+AdamW ──
echo ""
echo "████████████████████████████████████████████████████████████"
echo "  实验 C: Dice + Muon+AdamW"
echo "████████████████████████████████████████████████████████████"
python ${BASE_DIR}/scripts/finetune_sam2_lora.py \
    --loss-type dice \
    --optimizer muon_adam \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --val-every ${VAL_EVERY} \
    --patience 5 \
    --output ${BASE_DIR}/results/ablation_c

echo ""
echo "  测试 C..."
python ${BASE_DIR}/scripts/test_sam2_finetuned.py \
    --sam2-ckpt ${BASE_DIR}/results/ablation_c/checkpoints/best_inference.pth \
    --test-list ${TEST_LIST} \
    --output ${BASE_DIR}/results/ablation_c

# ── 汇总结果 ──
echo ""
echo "============================================================"
echo "  消融实验结果汇总"
echo "============================================================"

for exp in a b c; do
    OUTDIR="${BASE_DIR}/results/ablation_${exp}"
    JSON="${OUTDIR}/test_report.json"
    if [ -f "$JSON" ]; then
        IOU=$(python3 -c "import json; d=json.load(open('${JSON}')); print(f\"{d['metrics']['iou']:.4f}\")")
        NIOU=$(python3 -c "import json; d=json.load(open('${JSON}')); print(f\"{d['metrics']['niou']:.4f}\")")
        PD=$(python3 -c "import json; d=json.load(open('${JSON}')); print(f\"{d['metrics']['pd']:.4f}\")")
    else
        IOU="N/A"; NIOU="N/A"; PD="N/A"
    fi

    case $exp in
        a) DESC="Dice + AdamW (基线)" ;;
        b) DESC="Tversky + AdamW" ;;
        c) DESC="Dice + Muon+AdamW" ;;
    esac

    printf "  %-25s  IoU=%s  nIoU=%s  Pd=%s\n" "$DESC" "$IOU" "$NIOU" "$PD"
done

echo ""
echo "  结果已保存到 results/ablation_{a,b,c}/"
echo "============================================================"
