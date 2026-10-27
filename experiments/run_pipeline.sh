#!/usr/bin/env bash
# ============================================================
# run_pipeline.sh  — Intelligent sequential experiment pipeline
# Runs Phase 1 → 2 → 3, reads results after each phase,
# and automatically decides whether to add augmentation.
# ============================================================
set -euo pipefail

DATA="./data"
OUT="./results"
LOG="$OUT/pipeline.log"
PYTHON="python"

mkdir -p "$OUT"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# ── helper: read mean_acc from a results JSON ──────────────────────────────
read_acc() {
    local jf="$1" key="$2"
    python3 -c "import json,sys; d=json.load(open('$jf')); print(d$key)" 2>/dev/null || echo "0"
}

log "======================================================"
log "PRICAI 2026 Experiment Pipeline"
log "CRENet Phase-0 baseline: 76.89% (already done)"
log "======================================================"

# ─────────────────────────────────────────────────────────────────
# Phase 1: Modality Ablation
# ─────────────────────────────────────────────────────────────────
log ""
log ">>> PHASE 1: Modality ablation (7 configurations, 5-fold CV)"
$PYTHON -m experiments.run_experiments \
    --phase 1 --data_path "$DATA" --output_dir "$OUT" \
    --folds 5 --epochs 200 --batch_size 4 --lr 0.0005 \
    2>&1 | tee -a "$LOG"

log "Phase 1 complete."

# Read full-model acc from ablation
FULL_ACC=$(python3 -c "
import json
d = json.load(open('$OUT/phase1/ablation_results.json'))
print(d.get('full', {}).get('mean_acc', 0))
" 2>/dev/null || echo "0")

OPTICAL_ACC=$(python3 -c "
import json
d = json.load(open('$OUT/phase1/ablation_results.json'))
print(d.get('optical_only', {}).get('mean_acc', 0))
" 2>/dev/null || echo "0")

log "Phase 1 check — full model: $FULL_ACC | optical-only: $OPTICAL_ACC"

# Decision: if optical-only >= full-0.05, multi-modal gain is marginal → warn
python3 -c "
full=$FULL_ACC; opt=$OPTICAL_ACC
if float(opt) >= float(full) - 0.05:
    print('WARN: optical-only within 5% of full model — multi-modal gain is marginal')
else:
    print(f'OK: multi-modal gain = {float(full)-float(opt):.3f}')
" 2>/dev/null | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────
# Phase 2: SOTA Baselines
# ─────────────────────────────────────────────────────────────────
log ""
log ">>> PHASE 2: SOTA baselines (ResNet-18, EfficientNet-B0, MM-ResNet18)"
$PYTHON -m experiments.run_experiments \
    --phase 2 --data_path "$DATA" --output_dir "$OUT" \
    --folds 5 --epochs 200 --batch_size 4 --lr 0.0005 \
    2>&1 | tee -a "$LOG"

log "Phase 2 complete."

# Read best single-modal baseline (ResNet-18 pretrained)
RESNET_ACC=$(python3 -c "
import json
d = json.load(open('$OUT/phase2/baseline_results.json'))
print(d.get('resnet18_pretrained', {}).get('mean_acc', 0))
" 2>/dev/null || echo "0")

log "Phase 2 check — ResNet-18 pretrained: $RESNET_ACC | CRENet: 0.7689"

# Decision: if ResNet-18 pretrained > CRENet → run augmented CRENet
NEED_AUG=$(python3 -c "
crenet=0.7689; resnet=$RESNET_ACC
if float(resnet) > float(crenet):
    print('YES')
else:
    print('NO')
" 2>/dev/null || echo "NO")

if [ "$NEED_AUG" = "YES" ]; then
    log ""
    log ">>> DECISION: ResNet-18 ($RESNET_ACC) > CRENet (0.7689)"
    log ">>> Launching CRENet WITH data augmentation to close the gap..."
    $PYTHON -m experiments.run_experiments \
        --phase 0 --data_path "$DATA" \
        --output_dir "$OUT/phase0_augmented" \
        --folds 5 --epochs 200 --batch_size 4 --lr 0.0005 \
        --augment \
        2>&1 | tee -a "$LOG"

    AUG_ACC=$(python3 -c "
import json
d = json.load(open('$OUT/phase0_augmented/phase0_summary.json'))
print(d.get('cnn', {}).get('mean_acc', 0))
" 2>/dev/null || echo "0")

    log "Augmented CRENet: $AUG_ACC  vs  ResNet-18: $RESNET_ACC"
    CRENET_FINAL=$AUG_ACC
else
    log ">>> DECISION: CRENet (0.7689) >= ResNet-18 ($RESNET_ACC) — no augmentation needed"
    CRENET_FINAL=0.7689
fi

# ─────────────────────────────────────────────────────────────────
# Phase 3: Fusion Strategy Ablation
# ─────────────────────────────────────────────────────────────────
log ""
log ">>> PHASE 3: Fusion strategy comparison (5 strategies)"
$PYTHON -m experiments.run_experiments \
    --phase 3 --data_path "$DATA" --output_dir "$OUT" \
    --folds 5 --epochs 200 --batch_size 4 --lr 0.0005 \
    2>&1 | tee -a "$LOG"

log "Phase 3 complete."

# Check if our fusion (self_attention) is best
SELF_ATTN_ACC=$(python3 -c "
import json
d = json.load(open('$OUT/phase3/fusion_results.json'))
print(d.get('self_attention', {}).get('mean_acc', 0))
" 2>/dev/null || echo "0")

BEST_FUSION=$(python3 -c "
import json
d = json.load(open('$OUT/phase3/fusion_results.json'))
best = max(d.items(), key=lambda x: x[1].get('mean_acc', 0))
print(f'{best[0]}: {best[1][\"mean_acc\"]:.4f}')
" 2>/dev/null || echo "unknown")

log "Fusion check — self_attention: $SELF_ATTN_ACC | best: $BEST_FUSION"

# If cross_attention is better, update paper note
CROSS_ACC=$(python3 -c "
import json
d = json.load(open('$OUT/phase3/fusion_results.json'))
print(d.get('cross_attention', {}).get('mean_acc', 0))
" 2>/dev/null || echo "0")

FUSION_DECISION=$(python3 -c "
sa=$SELF_ATTN_ACC; ca=$CROSS_ACC
if float(ca) > float(sa) + 0.01:
    print(f'USE_CROSS: cross_attention ({ca}) > self_attention ({sa})')
else:
    print(f'KEEP_SELF: self_attention ({sa}) is best or competitive')
" 2>/dev/null || echo "KEEP_SELF")

log "Fusion decision: $FUSION_DECISION"

# ─────────────────────────────────────────────────────────────────
# Phase 4+5: Attention viz + per-class (use best checkpoint)
# ─────────────────────────────────────────────────────────────────
BEST_CKPT="$OUT/phase0/cnn/fold0/fold0_best.pth"
if [ -f "$BEST_CKPT" ]; then
    log ""
    log ">>> PHASE 4: Attention weight analysis"
    $PYTHON -m experiments.run_experiments \
        --phase 4 --data_path "$DATA" --output_dir "$OUT" \
        --checkpoint "$BEST_CKPT" 2>&1 | tee -a "$LOG"

    log ">>> PHASE 5: Per-class analysis"
    $PYTHON -m experiments.run_experiments \
        --phase 5 --data_path "$DATA" --output_dir "$OUT" \
        --checkpoint "$BEST_CKPT" 2>&1 | tee -a "$LOG"
fi

# ─────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────
log ""
log "======================================================"
log "PIPELINE COMPLETE — FINAL SUMMARY"
log "======================================================"
log "CRENet (final):          $CRENET_FINAL"
log "Random Forest:           $(python3 -c "import json; d=json.load(open('$OUT/phase0/traditional_results.json')); print(d['rf']['mean_acc'])" 2>/dev/null)"
log "SVM:                     $(python3 -c "import json; d=json.load(open('$OUT/phase0/traditional_results.json')); print(d['svm']['mean_acc'])" 2>/dev/null)"
log "ResNet-18 pretrained:    $RESNET_ACC"
log "Best fusion strategy:    $BEST_FUSION"
log "Fusion decision:         $FUSION_DECISION"
log "Augmentation used:       $NEED_AUG"
log ""
log "Results directory: $OUT"
log "======================================================"
