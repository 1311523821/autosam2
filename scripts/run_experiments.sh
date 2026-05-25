#!/bin/bash
# ============================================================
# 引导头训练实验脚本 - 基于消融实验结果优化
# 运行方式: nohup bash scripts/run_experiments.sh > train.log 2>&1 &
# 预计时间: 约4-6小时
# ============================================================

# 不使用 set -e，允许单个实验失败后继续
# set -e

# 公共参数
export PYTHONPATH="/root/autosam2:/root/e2e:$PYTHONPATH"
DATA_ROOT="/root/DataBscan"
TRAIN_LIST="/root/e2e/train1.txt"
TEST_LIST="/root/e2e/test1.txt"
OUTPUT_BASE="checkpoints/experiments"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${OUTPUT_BASE}/experiment_log_${TIMESTAMP}.txt"

# 创建输出目录
mkdir -p ${OUTPUT_BASE}

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a ${LOG_FILE}
}

log "============================================"
log "引导头训练消融实验开始"
log "架构: 纯 STSF (Swin 已验证无效)"
log "分辨率: 256x256"
log "评估阈值: 5像素 (正确比例)"
log "实验数量: 6组"
log "测试集: ${TEST_LIST}"
log "============================================"

# ============================================================
# A组: 学习率对比 (Exp 1-3)
# ============================================================

# Exp 1: 基线 - AdamW + lr=5e-4
log "Exp 1: 基线 - AdamW + lr=5e-4"
python train_guidance.py \
    --kfold 5 \
    --epochs 30 \
    --batch-size 8 \
    --embed-dim 32 \
    --hidden-dim 16 \
    --optimizer adamw \
    --heatmap-sigma 10.0 \
    --img-size 256 256 \
    --threshold-radius 5 \
    --seed 42 \
    --lr 5e-4 \
    --accum-steps 2 \
    --patience 15 \
    --warmup-epochs 3 \
    --weight-decay 1e-4 \
    --loss-type heatmap_focal \
    --focal-alpha 2 \
    --focal-beta 4 \
    --output ${OUTPUT_BASE}/exp1_lr5e-4 \
    --data-root ${DATA_ROOT} \
    --train-list ${TRAIN_LIST} \
    --test-list ${TEST_LIST} \
    2>&1 | tee -a ${LOG_FILE}
log "Exp 1 完成"

# Exp 2: 更大学习率 1e-3
log "Exp 2: AdamW + lr=1e-3"
python train_guidance.py \
    --kfold 5 \
    --epochs 30 \
    --batch-size 8 \
    --embed-dim 32 \
    --hidden-dim 16 \
    --optimizer adamw \
    --heatmap-sigma 10.0 \
    --img-size 256 256 \
    --threshold-radius 5 \
    --seed 42 \
    --lr 1e-3 \
    --accum-steps 2 \
    --patience 15 \
    --warmup-epochs 3 \
    --weight-decay 1e-4 \
    --loss-type heatmap_focal \
    --focal-alpha 2 \
    --focal-beta 4 \
    --output ${OUTPUT_BASE}/exp2_lr1e-3 \
    --data-root ${DATA_ROOT} \
    --train-list ${TRAIN_LIST} \
    --test-list ${TEST_LIST} \
    2>&1 | tee -a ${LOG_FILE}
log "Exp 2 完成"

# Exp 3: 更小学习率 1e-4
log "Exp 3: AdamW + lr=1e-4"
python train_guidance.py \
    --kfold 5 \
    --epochs 30 \
    --batch-size 8 \
    --embed-dim 32 \
    --hidden-dim 16 \
    --optimizer adamw \
    --heatmap-sigma 10.0 \
    --img-size 256 256 \
    --threshold-radius 5 \
    --lr 1e-4 \
    --accum-steps 2 \
    --patience 15 \
    --warmup-epochs 3 \
    --weight-decay 1e-4 \
    --loss-type heatmap_focal \
    --focal-alpha 2 \
    --focal-beta 4 \
    --output ${OUTPUT_BASE}/exp3_lr1e-4 \
    --data-root ${DATA_ROOT} \
    --train-list ${TRAIN_LIST} \
    --test-list ${TEST_LIST} \
    2>&1 | tee -a ${LOG_FILE}
log "Exp 3 完成"

# ============================================================
# B组: 损失函数对比 (Exp 4-5)
# ============================================================

# Exp 4: MSE损失
log "Exp 4: AdamW + MSE Loss"
python train_guidance.py \
    --kfold 5 \
    --epochs 30 \
    --batch-size 8 \
    --embed-dim 32 \
    --hidden-dim 16 \
    --optimizer adamw \
    --heatmap-sigma 10.0 \
    --img-size 256 256 \
    --threshold-radius 5 \
    --seed 42 \
    --lr 5e-4 \
    --accum-steps 2 \
    --patience 15 \
    --warmup-epochs 3 \
    --weight-decay 1e-4 \
    --loss-type mse \
    --output ${OUTPUT_BASE}/exp4_mse \
    --data-root ${DATA_ROOT} \
    --train-list ${TRAIN_LIST} \
    --test-list ${TEST_LIST} \
    2>&1 | tee -a ${LOG_FILE}
log "Exp 4 完成"

# Exp 5: Focal Loss + 更大alpha
log "Exp 5: AdamW + Focal(alpha=4)"
python train_guidance.py \
    --kfold 5 \
    --epochs 30 \
    --batch-size 8 \
    --embed-dim 32 \
    --hidden-dim 16 \
    --optimizer adamw \
    --heatmap-sigma 10.0 \
    --img-size 256 256 \
    --threshold-radius 5 \
    --seed 42 \
    --lr 5e-4 \
    --accum-steps 2 \
    --patience 15 \
    --warmup-epochs 3 \
    --weight-decay 1e-4 \
    --loss-type heatmap_focal \
    --focal-alpha 4 \
    --focal-beta 4 \
    --output ${OUTPUT_BASE}/exp5_focal_a4 \
    --data-root ${DATA_ROOT} \
    --train-list ${TRAIN_LIST} \
    --test-list ${TEST_LIST} \
    2>&1 | tee -a ${LOG_FILE}
log "Exp 5 完成"

# ============================================================
# C组: 模型容量对比 (Exp 6)
# ============================================================

# Exp 6: 更大模型 embed_dim=64
log "Exp 6: AdamW + embed_dim=64"
python train_guidance.py \
    --kfold 5 \
    --epochs 30 \
    --batch-size 8 \
    --embed-dim 64 \
    --hidden-dim 32 \
    --optimizer adamw \
    --heatmap-sigma 10.0 \
    --img-size 256 256 \
    --threshold-radius 5 \
    --seed 42 \
    --lr 5e-4 \
    --accum-steps 2 \
    --patience 15 \
    --warmup-epochs 3 \
    --weight-decay 1e-4 \
    --loss-type heatmap_focal \
    --focal-alpha 2 \
    --focal-beta 4 \
    --output ${OUTPUT_BASE}/exp6_embed64 \
    --data-root ${DATA_ROOT} \
    --train-list ${TRAIN_LIST} \
    --test-list ${TEST_LIST} \
    2>&1 | tee -a ${LOG_FILE}
log "Exp 6 完成"

# ============================================================
# 汇总结果
# ============================================================
log "============================================"
log "所有实验完成！正在汇总结果..."
log "============================================"

# 提取各实验的最佳结果
RESULT_FILE="${OUTPUT_BASE}/results_summary_${TIMESTAMP}.txt"
echo "============================================" > ${RESULT_FILE}
echo "引导头训练消融实验结果汇总" >> ${RESULT_FILE}
echo "时间: ${TIMESTAMP}" >> ${RESULT_FILE}
echo "架构: 纯 STSF (无 Swin)" >> ${RESULT_FILE}
echo "分辨率: 256x256" >> ${RESULT_FILE}
echo "评估阈值: 5像素" >> ${RESULT_FILE}
echo "============================================" >> ${RESULT_FILE}
echo "" >> ${RESULT_FILE}

# 解析并格式化结果
for exp_dir in ${OUTPUT_BASE}/exp*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename $exp_dir)
        echo "----------------------------------------" >> ${RESULT_FILE}
        echo "实验: $exp_name" >> ${RESULT_FILE}
        echo "----------------------------------------" >> ${RESULT_FILE}

        # 查找嵌套的模型目录 (enc(...))
        model_dir=$(find "$exp_dir" -maxdepth 1 -type d -name "enc*" | head -n 1)

        if [ -n "$model_dir" ]; then
            # K-Fold 结果
            if [ -f "$model_dir/kfold_results.txt" ]; then
                echo "[验证集 K-Fold 结果]" >> ${RESULT_FILE}
                cat "$model_dir/kfold_results.txt" >> ${RESULT_FILE}
            fi

            # 测试集结果 - 查找最佳Fold的结果
            best_fold=""
            best_hr=0
            for fold_dir in "$model_dir"/fold_*; do
                if [ -d "$fold_dir" ] && [ -f "$fold_dir/test_results.txt" ]; then
                    hr=$(grep "Hit Rate:" "$fold_dir/test_results.txt" | awk '{print $NF}' | tr -d '%')
                    if [ -n "$hr" ] && (( $(echo "$hr > $best_hr" | bc -l 2>/dev/null || echo 0) )); then
                        best_hr=$hr
                        best_fold=$fold_dir
                    fi
                fi
            done

            if [ -n "$best_fold" ] && [ -f "$best_fold/test_results.txt" ]; then
                echo "" >> ${RESULT_FILE}
                echo "[测试集评估结果] (最佳Fold: $(basename $best_fold))" >> ${RESULT_FILE}
                cat "$best_fold/test_results.txt" >> ${RESULT_FILE}
            fi
        fi
        echo "" >> ${RESULT_FILE}
    fi
done

# 生成对比表格
echo "" >> ${RESULT_FILE}
echo "============================================" >> ${RESULT_FILE}
echo "快速对比表" >> ${RESULT_FILE}
echo "============================================" >> ${RESULT_FILE}
echo "Exp | 验证集Hit Rate | 测试集Hit Rate" >> ${RESULT_FILE}
echo "----|----------------|----------------" >> ${RESULT_FILE}

for exp_dir in ${OUTPUT_BASE}/exp*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename $exp_dir)
        val_hr=""
        test_hr=""

        # 查找嵌套的模型目录
        model_dir=$(find "$exp_dir" -maxdepth 1 -type d -name "enc*" | head -n 1)

        if [ -n "$model_dir" ]; then
            # 验证集Hit Rate (Hit Rate@10)
            if [ -f "$model_dir/kfold_results.txt" ]; then
                val_hr=$(grep "平均 Hit Rate@10" "$model_dir/kfold_results.txt" | awk '{print $4}')
            fi

            # 测试集Hit Rate (Hit Rate@5, 从kfold_results.txt汇总)
            if [ -f "$model_dir/kfold_results.txt" ]; then
                test_hr=$(grep "平均 Hit Rate@5" "$model_dir/kfold_results.txt" | awk '{print $4}')
            fi
        fi

        echo "$exp_name | ${val_hr:-N/A} | ${test_hr:-N/A}" >> ${RESULT_FILE}
    fi
done

log "结果已汇总到: ${RESULT_FILE}"
log "实验结束！"

# 打印汇总结果
cat ${RESULT_FILE}
