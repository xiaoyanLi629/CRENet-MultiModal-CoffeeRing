#!/usr/bin/env bash
# ============================================================
# run_all.sh — Run all PRICAI 2026 experiment phases in sequence
# ============================================================
# Usage:
#   chmod +x experiments/run_all.sh
#   ./experiments/run_all.sh                   # full run (5-fold, 200 epochs)
#   ./experiments/run_all.sh --debug           # quick code test (2-fold, 2 epochs)
#
# All results are saved to ./experiment_results/phase{N}/
# ============================================================

set -euo pipefail

DATA_PATH="./data"
OUTPUT_DIR="./results"
EXTRA_ARGS="${@}"   # pass --debug or any extra flags through

PYTHON="python"

echo "======================================================"
echo "PRICAI 2026 Experiment Suite"
echo "Data:   $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Extra:  $EXTRA_ARGS"
echo "======================================================"

# ── Phase 0: Base model ────────────────────────────────────────────────────
echo ""
echo ">>> Phase 0: Base model (CNN EDS encoder, self-attention)"
$PYTHON -m experiments.run_experiments \
    --phase 0 \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    $EXTRA_ARGS

# Set checkpoint to fold 0 best model from Phase 0
CHECKPOINT="$OUTPUT_DIR/phase0/fold0/fold0_best.pth"

# ── Phase 1: Modality ablation ─────────────────────────────────────────────
echo ""
echo ">>> Phase 1: Modality ablation"
$PYTHON -m experiments.run_experiments \
    --phase 1 \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    $EXTRA_ARGS

# ── Phase 2: SOTA baselines ────────────────────────────────────────────────
echo ""
echo ">>> Phase 2: SOTA baselines"
$PYTHON -m experiments.run_experiments \
    --phase 2 \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    $EXTRA_ARGS

# ── Phase 3: Fusion strategy ablation ─────────────────────────────────────
echo ""
echo ">>> Phase 3: Fusion strategy ablation"
$PYTHON -m experiments.run_experiments \
    --phase 3 \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    $EXTRA_ARGS

# ── Phase 4: Attention visualisation ──────────────────────────────────────
echo ""
echo ">>> Phase 4: Attention weight analysis"
$PYTHON -m experiments.run_experiments \
    --phase 4 \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    $EXTRA_ARGS

# ── Phase 5: Per-class analysis ────────────────────────────────────────────
echo ""
echo ">>> Phase 5: Per-class CNN performance"
$PYTHON -m experiments.run_experiments \
    --phase 5 \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    $EXTRA_ARGS

echo ""
echo "======================================================"
echo "All phases complete. Results in: $OUTPUT_DIR"
echo "======================================================"
