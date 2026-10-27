"""
experiments/run_experiments.py
==============================
Unified experiment runner for all PRICAI 2026 phases.

Usage examples
--------------
# Phase 0 — base model with CNN EDS encoder (full 5-fold CV):
  python experiments/run_experiments.py --phase 0 --data_path ./data

# Phase 1 — modality ablation:
  python experiments/run_experiments.py --phase 1 --data_path ./data

# Phase 2 — SOTA baselines:
  python experiments/run_experiments.py --phase 2 --data_path ./data

# Phase 3 — fusion strategy ablation:
  python experiments/run_experiments.py --phase 3 --data_path ./data

# Phase 4 — attention weight visualisation:
  python experiments/run_experiments.py --phase 4 --data_path ./data \
      --checkpoint ./results/phase0/fold0/fold0_best.pth

# Phase 5 — per-class analysis:
  python experiments/run_experiments.py --phase 5 --data_path ./data \
      --checkpoint ./results/phase0/fold0/fold0_best.pth

# Quick code test (no GPU, 2 epochs, 2 samples/class, 2 folds):
  python experiments/run_experiments.py --phase 0 --data_path ./data --debug
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for server
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.models import (
    CRENet,
    CRENetPretrained,
    CRENetEfficient,
    EarlyFusionCNN,
    ResNet18Baseline,
    EfficientNetBaseline,
    MultiModalResNetBaseline,
    FUSION_MODES,
)
from experiments.train_utils import (
    load_dataset,
    cross_validate,
    cross_validate_pretrained,
    cross_validate_traditional,
    compute_esi_clean,
    compute_per_class_metrics,
    save_results,
    _predict,
    CONDITION_NAMES,
)
from ml_framework import CoffeeRingDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Shared config builder
# ─────────────────────────────────────────────────────────────────────────────

def get_train_kwargs(args):
    return dict(
        n_folds    = args.folds,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        patience   = 10 if args.debug else 30,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0 — Base model (CNN EDS encoder, self-attention fusion)
# ─────────────────────────────────────────────────────────────────────────────

def run_phase0(args, device, optical, sem, eds, labels):
    """
    CLEAN baseline run: CNN + RF + SVM + ESI, all with proper k-fold CV.
    No data leakage, no manipulation.

    Replaces the corrupted production_results/ entirely.
    """
    print("\n" + "="*70)
    print("PHASE 0: Clean baseline (CNN + RF + SVM + ESI)")
    print("  • Strict k-fold CV — zero test data leakage")
    print("  • ESI computed without condition-specific multipliers")
    print("="*70)

    out_dir = os.path.join(args.output_dir, "phase0")
    os.makedirs(out_dir, exist_ok=True)

    # ── CNN ───────────────────────────────────────────────────────────────────
    print("\n[1/3] Training CRENet (self-attention, CNN EDS encoder)...")

    def model_fn():
        return CRENet(num_classes=9, fusion='self_attention')

    m = model_fn()
    print(f"  Parameters: {m.count_parameters():,}")
    del m

    cnn_results = cross_validate(
        model_fn, optical, sem, eds, labels, device,
        output_dir=os.path.join(out_dir, 'cnn'),
        augment=args.augment,
        **get_train_kwargs(args),
    )
    cnn_results['per_class_acc'] = compute_per_class_metrics(
        cnn_results['all_preds'], cnn_results['all_truths']
    )
    _plot_confusion_matrix(
        cnn_results['all_preds'], cnn_results['all_truths'],
        title='CNN Confusion Matrix',
        save_path=os.path.join(out_dir, 'confusion_matrix_cnn.png'),
    )
    _plot_training_curves(
        cnn_results['per_fold'][0]['history'],
        save_path=os.path.join(out_dir, 'training_curve_fold0.png'),
    )
    save_results(cnn_results, os.path.join(out_dir, 'cnn_results.json'))
    _print_summary("CNN", cnn_results)

    # ── RF + SVM ──────────────────────────────────────────────────────────────
    print("\n[2/3] Training RF and SVM (hand-crafted morphological features)...")
    tune = not args.debug   # skip GridSearchCV in debug mode

    trad_results = cross_validate_traditional(
        optical, labels,
        n_folds=get_train_kwargs(args)['n_folds'],
        tune_svm=tune,
        output_dir=os.path.join(out_dir, 'traditional'),
    )

    _plot_confusion_matrix(
        trad_results['rf']['all_preds'], trad_results['rf']['all_truths'],
        title='Random Forest Confusion Matrix',
        save_path=os.path.join(out_dir, 'confusion_matrix_rf.png'),
    )
    save_results(trad_results, os.path.join(out_dir, 'traditional_results.json'))

    print(f"\nRF:  acc={trad_results['rf']['mean_acc']:.4f}±"
          f"{trad_results['rf']['std_acc']:.4f}")
    print(f"SVM: acc={trad_results['svm']['mean_acc']:.4f}±"
          f"{trad_results['svm']['std_acc']:.4f}")

    # ── Ensemble (CNN + RF majority vote, per-fold) ───────────────────────────
    cnn_p = cnn_results['all_preds']
    rf_p  = trad_results['rf']['all_preds']
    truths = cnn_results['all_truths']
    ensemble_preds = [
        max(set([c, r]), key=[c, r].count)  # majority of 2 → prefer CNN on tie
        for c, r in zip(cnn_p, rf_p)
    ]
    ens_acc = accuracy_score(truths, ensemble_preds)
    ens_f1  = f1_score(truths, ensemble_preds, average='weighted', zero_division=0)
    print(f"Ensemble (CNN+RF majority): acc={ens_acc:.4f}, f1={ens_f1:.4f}")

    # ── Clean ESI ─────────────────────────────────────────────────────────────
    print("\n[3/3] Computing clean ESI (no manipulation)...")
    esi_results = compute_esi_clean(eds, labels, alpha=0.6, beta=0.4)
    save_results(esi_results, os.path.join(out_dir, 'esi_results_clean.json'))
    _plot_esi_bars(esi_results, save_path=os.path.join(out_dir, 'esi_clean.png'))

    # ── Combined summary ──────────────────────────────────────────────────────
    combined = {
        'cnn':      {'mean_acc': cnn_results['mean_acc'],
                     'std_acc':  cnn_results['std_acc'],
                     'mean_f1':  cnn_results['mean_f1'],
                     'std_f1':   cnn_results['std_f1']},
        'rf':       {'mean_acc': trad_results['rf']['mean_acc'],
                     'std_acc':  trad_results['rf']['std_acc'],
                     'mean_f1':  trad_results['rf']['mean_f1'],
                     'std_f1':   trad_results['rf']['std_f1']},
        'svm':      {'mean_acc': trad_results['svm']['mean_acc'],
                     'std_acc':  trad_results['svm']['std_acc'],
                     'mean_f1':  trad_results['svm']['mean_f1'],
                     'std_f1':   trad_results['svm']['std_f1']},
        'ensemble': {'acc': ens_acc, 'f1': ens_f1},
        'esi':      esi_results,
        'note':     'All results from clean 5-fold CV. No data leakage. '
                    'ESI computed without condition-specific multipliers.',
    }
    save_results(combined, os.path.join(out_dir, 'phase0_summary.json'))

    # Summary table
    print(f"\n{'='*60}")
    print(f"PHASE 0 CLEAN RESULTS SUMMARY")
    print(f"{'─'*60}")
    print(f"{'Model':<20} {'Acc (mean±std)':>18} {'F1 (mean±std)':>16}")
    print(f"{'─'*60}")
    for name, key in [('CNN (ours)',    'cnn'),
                       ('Random Forest', 'rf'),
                       ('SVM (tuned)',   'svm')]:
        r = combined[key]
        print(f"{name:<20} {r['mean_acc']:.4f} ± {r['std_acc']:.4f}   "
              f"{r['mean_f1']:.4f} ± {r['std_f1']:.4f}")
    print(f"{'Ensemble (CNN+RF)':<20} {ens_acc:.4f}              "
          f"{ens_f1:.4f}")
    print(f"{'='*60}")

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Modality Ablation
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_CONFIGS = {
    'optical_only':  {'optical': True,  'sem': False, 'eds': False},
    'sem_only':      {'optical': False, 'sem': True,  'eds': False},
    'eds_only':      {'optical': False, 'sem': False, 'eds': True},
    'optical_sem':   {'optical': True,  'sem': True,  'eds': False},
    'optical_eds':   {'optical': True,  'sem': False, 'eds': True},
    'sem_eds':       {'optical': False, 'sem': True,  'eds': True},
    'full':          {'optical': True,  'sem': True,  'eds': True},
}


def run_phase1(args, device, optical, sem, eds, labels):
    """
    Modality ablation: train 7 configurations, each with k-fold CV.
    """
    print("\n" + "="*70)
    print("PHASE 1: Modality ablation study")
    print("="*70)

    out_dir = os.path.join(args.output_dir, "phase1")
    all_results = {}

    for config_name, mask in ABLATION_CONFIGS.items():
        print(f"\n--- Ablation: {config_name} ---")
        print(f"    Mask: {mask}")

        def model_fn(m=mask):
            return CRENet(num_classes=9, fusion='self_attention')

        config_dir = os.path.join(out_dir, config_name)
        results = cross_validate(
            model_fn, optical, sem, eds, labels, device,
            modality_mask=mask,
            output_dir=config_dir,
            **get_train_kwargs(args),
        )
        all_results[config_name] = {
            'mask': mask,
            'mean_acc': results['mean_acc'],
            'std_acc':  results['std_acc'],
            'mean_f1':  results['mean_f1'],
            'std_f1':   results['std_f1'],
        }
        print(f"  {config_name}: acc={results['mean_acc']:.4f}±{results['std_acc']:.4f}")

    save_results(all_results, os.path.join(out_dir, 'ablation_results.json'))
    _plot_ablation(all_results, save_path=os.path.join(out_dir, 'ablation_plot.png'))
    _print_ablation_table(all_results)
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — SOTA Baselines
# ─────────────────────────────────────────────────────────────────────────────

def run_phase2(args, device, optical, sem, eds, labels):
    """
    Train 5 SOTA baselines with k-fold CV.
    """
    print("\n" + "="*70)
    print("PHASE 2: SOTA baseline comparison")
    print("="*70)

    out_dir = os.path.join(args.output_dir, "phase2")

    # 3 baselines:
    # 1. ResNet-18 pretrained  — single-modal SOTA representative
    # 2. EfficientNet-B0       — lightweight single-modal SOTA
    # 3. Multi-modal ResNet-18 — heavyweight multi-modal comparison
    # (ResNet-18 scratch and ViT-Small removed:
    #   scratch vs pretrained is a known result not worth demonstrating;
    #   ViT needs far more data than 225 samples to be a fair comparison.)
    baselines = {
        'resnet18_pretrained': lambda: ResNet18Baseline(pretrained=True),
        'efficientnet_b0':     lambda: EfficientNetBaseline(pretrained=True),
        'multimodal_resnet18': lambda: MultiModalResNetBaseline(pretrained=False),
    }

    all_results = {}
    for name, model_fn in baselines.items():
        print(f"\n--- Baseline: {name} ---")
        try:
            m = model_fn()
            print(f"    Parameters: {m.count_parameters():,}")
            del m
        except Exception as e:
            print(f"    [WARN] Could not instantiate {name}: {e}")
            continue

        results = cross_validate(
            model_fn, optical, sem, eds, labels, device,
            output_dir=os.path.join(out_dir, name),
            **get_train_kwargs(args),
        )
        all_results[name] = {
            'mean_acc': results['mean_acc'],
            'std_acc':  results['std_acc'],
            'mean_f1':  results['mean_f1'],
            'std_f1':   results['std_f1'],
        }
        print(f"  {name}: acc={results['mean_acc']:.4f}±{results['std_acc']:.4f}")

    save_results(all_results, os.path.join(out_dir, 'baseline_results.json'))
    _print_baseline_table(all_results)
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Fusion Strategy Ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_phase3(args, device, optical, sem, eds, labels):
    """
    Compare 5 fusion strategies: early, late_concat, self_attention,
    cross_attention, gated.
    """
    print("\n" + "="*70)
    print("PHASE 3: Fusion strategy comparison")
    print("="*70)

    out_dir = os.path.join(args.output_dir, "phase3")
    all_results = {}

    # Attention-based strategies (use CRENet)
    for fusion in FUSION_MODES:
        print(f"\n--- Fusion: {fusion} ---")

        def model_fn(f=fusion):
            return CRENet(num_classes=9, fusion=f)

        results = cross_validate(
            model_fn, optical, sem, eds, labels, device,
            output_dir=os.path.join(out_dir, fusion),
            **get_train_kwargs(args),
        )
        all_results[fusion] = {
            'mean_acc': results['mean_acc'],
            'std_acc':  results['std_acc'],
            'mean_f1':  results['mean_f1'],
            'std_f1':   results['std_f1'],
        }

    # Early fusion (separate model class)
    print("\n--- Fusion: early_fusion ---")
    def early_fn():
        return EarlyFusionCNN(num_classes=9)

    results = cross_validate(
        early_fn, optical, sem, eds, labels, device,
        output_dir=os.path.join(out_dir, 'early_fusion'),
        **get_train_kwargs(args),
    )
    all_results['early_fusion'] = {
        'mean_acc': results['mean_acc'],
        'std_acc':  results['std_acc'],
        'mean_f1':  results['mean_f1'],
        'std_f1':   results['std_f1'],
    }

    save_results(all_results, os.path.join(out_dir, 'fusion_results.json'))
    _print_fusion_table(all_results)
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Attention Weight Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def run_phase4(args, device, optical, sem, eds, labels):
    """
    Load a trained checkpoint, extract attention weights on the full dataset,
    and generate per-condition heatmaps.

    Requires --checkpoint pointing to a saved .pth file from Phase 0.
    """
    print("\n" + "="*70)
    print("PHASE 4: Attention weight analysis")
    print("="*70)

    if not args.checkpoint or not os.path.exists(args.checkpoint):
        print(f"[WARN] --checkpoint not provided or does not exist: {args.checkpoint}")
        print("       Running Phase 0 first to generate a checkpoint, or pass --checkpoint.")
        return {}

    out_dir = os.path.join(args.output_dir, "phase4")
    os.makedirs(out_dir, exist_ok=True)

    model = CRENet(num_classes=9, fusion='self_attention').to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Collect attention weights per condition
    attn_by_condition = {i: [] for i in range(9)}

    from ml_framework import CoffeeRingDataset
    from torch.utils.data import DataLoader

    dataset = CoffeeRingDataset(optical, sem, eds, labels)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            opt_b = batch['optical'].to(device)
            sem_b = batch['sem'].to(device)
            eds_b = batch['eds'].to(device)
            y_b   = batch['label'].view(-1)

            _, attn_w = model(opt_b, sem_b, eds_b, return_attention=True)
            # attn_w: (batch, num_heads, 2, 2)  — averaged over heads
            # [b, h, 0, 0] = optical→optical, [b, h, 0, 1] = optical→sem
            # [b, h, 1, 0] = sem→optical,     [b, h, 1, 1] = sem→sem
            if attn_w is not None:
                # attn_w from nn.MultiheadAttention is (B, seq_len, seq_len)
                # already averaged over heads; shape (B, 2, 2) for our 2-token input
                attn_np = attn_w.cpu().numpy()  # (B, 2, 2)
                for i, cls in enumerate(y_b.tolist()):
                    attn_by_condition[int(cls)].append(attn_np[i])

    # Average per condition
    condition_attn = {}
    for cond_id in range(9):
        if attn_by_condition[cond_id]:
            condition_attn[CONDITION_NAMES[cond_id]] = np.mean(
                attn_by_condition[cond_id], axis=0
            ).tolist()

    # ── Plot heatmap ──────────────────────────────────────────────────────────
    cond_labels = [n.split('_')[1] for n in CONDITION_NAMES]  # A–I
    # For each condition, optical→optical weight (how much optical attends to itself)
    opt_self  = [condition_attn.get(CONDITION_NAMES[i], [[0,0],[0,0]])[0][0] for i in range(9)]
    opt_to_sem= [condition_attn.get(CONDITION_NAMES[i], [[0,0],[0,0]])[0][1] for i in range(9)]
    sem_self  = [condition_attn.get(CONDITION_NAMES[i], [[0,0],[0,0]])[1][1] for i in range(9)]
    sem_to_opt= [condition_attn.get(CONDITION_NAMES[i], [[0,0],[0,0]])[1][0] for i in range(9)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, vals, title in zip(
        axes,
        [opt_self, sem_self],
        ['Optical → Self Attention Weight', 'SEM → Self Attention Weight'],
    ):
        bars = ax.bar(cond_labels, vals, color='steelblue', edgecolor='white')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Environmental Condition')
        ax.set_ylabel('Mean Attention Weight')
        ax.set_ylim(0, 1)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'attention_weights.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Attention heatmap saved → {out_dir}/attention_weights.png")

    save_results(condition_attn, os.path.join(out_dir, 'attention_weights.json'))
    return condition_attn


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Per-class CNN Performance Analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_phase5(args, device, optical, sem, eds, labels):
    """
    Load Phase 0 checkpoint; compute CNN per-condition accuracy and generate
    CNN confusion matrix figure.
    """
    print("\n" + "="*70)
    print("PHASE 5: Per-class CNN performance analysis")
    print("="*70)

    if not args.checkpoint or not os.path.exists(args.checkpoint):
        print(f"[WARN] --checkpoint not provided. Running inference on all data "
              f"(train=test, for diagnostics only).")

    out_dir = os.path.join(args.output_dir, "phase5")
    os.makedirs(out_dir, exist_ok=True)

    model = CRENet(num_classes=9, fusion='self_attention').to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.checkpoint}")
    model.eval()

    from ml_framework import CoffeeRingDataset
    from torch.utils.data import DataLoader

    dataset = CoffeeRingDataset(optical, sem, eds, labels)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False)

    preds, truths = [], []
    with torch.no_grad():
        for batch in loader:
            opt_b = batch['optical'].to(device)
            sem_b = batch['sem'].to(device)
            eds_b = batch['eds'].to(device)
            y_b   = batch['label'].view(-1)

            logits = model(opt_b, sem_b, eds_b)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            truths.extend(y_b.tolist())

    per_class = compute_per_class_metrics(preds, truths)

    # Confusion matrix
    _plot_confusion_matrix(
        preds, truths,
        title='CNN Per-Class Confusion Matrix',
        save_path=os.path.join(out_dir, 'confusion_matrix_cnn.png'),
    )

    # Print table
    print("\nPer-class CNN accuracy:")
    print(f"{'Condition':<14} {'CNN Acc':>8}")
    print('-' * 24)
    for name, acc in per_class.items():
        print(f"{name:<14} {acc:>8.4f}")

    results = {'per_class_acc': per_class, 'preds': preds, 'truths': truths}
    save_results(results, os.path.join(out_dir, 'perclass_results.json'))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — CRENetPretrained (pretrained EfficientNet-B0 + late_concat)
# ─────────────────────────────────────────────────────────────────────────────

def run_phase6(args, device, optical, sem, eds, labels):
    """
    Train CRENetEfficient: pretrained EfficientNet-B0 (optical) +
    scratch CNNs (SEM, EDS) + late_concat fusion.

    Motivation: Phase 6a (CRENetPretrained with two competing pretrained
    backbones) gave worse results than scratch CRENet. Root cause:
      1. Adapted SEM pretrained weights (RGB→grayscale) don't transfer well.
      2. Two backbone optimisers interfere during fine-tuning.
    Solution: use proven pretrained EfficientNet for optical only; keep
    SEM and EDS as scratch CNNs. Fine-tune all params with LR=5e-4,
    identical to the Phase 2 setup that achieved 92% on optical alone.
    """
    print("\n" + "="*70)
    print("PHASE 6: CRENetEfficient")
    print("  optical → pretrained EfficientNet-B0 (1280-dim)")
    print("  SEM     → scratch 5-layer CNN (256-dim)")
    print("  EDS     → scratch 3-layer CNN (128-dim)")
    print("  fusion  → late_concat (1664-dim) → FC → 9")
    print("  LR=5e-4 uniform (same as Phase 2 which gave 92%)")
    print("="*70)

    out_dir = os.path.join(args.output_dir, "phase6")
    os.makedirs(out_dir, exist_ok=True)

    def model_fn():
        return CRENetEfficient(num_classes=9)

    m = model_fn()
    print(f"  Total parameters: {m.count_parameters():,}")
    del m

    results = cross_validate(
        model_fn, optical, sem, eds, labels, device,
        output_dir = os.path.join(out_dir, 'crenet_efficient'),
        **get_train_kwargs(args),
    )

    results['per_class_acc'] = compute_per_class_metrics(
        results['all_preds'], results['all_truths'])

    _plot_confusion_matrix(
        results['all_preds'], results['all_truths'],
        title='CRENetEfficient Confusion Matrix',
        save_path=os.path.join(out_dir, 'confusion_matrix.png'),
    )
    _plot_training_curves(
        results['per_fold'][0]['history'],
        save_path=os.path.join(out_dir, 'training_curve_fold0.png'),
    )
    save_results(results, os.path.join(out_dir, 'results.json'))

    print(f"\n{'='*60}")
    print(f"PHASE 6 RESULT — CRENetEfficient")
    print(f"  Acc : {results['mean_acc']:.4f} ± {results['std_acc']:.4f}")
    print(f"  F1  : {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
    print(f"\nComparison:")
    print(f"  EfficientNet-B0 optical-only (Phase 2) : 92.00%")
    print(f"  CRENet scratch   3-modal (Phase 0)     : 76.89%")
    print(f"  CRENetEfficient  3-modal (Phase 6)     : {results['mean_acc']*100:.2f}%")
    if results['mean_acc'] >= 0.92:
        print("  ✓ Multi-modal MATCHES or EXCEEDS single-modal pretrained!")
    elif results['mean_acc'] >= 0.88:
        print("  ✓ Multi-modal COMPETITIVE — gap within fine-tuning variance.")
    else:
        print("  ⚠ Below EfficientNet-B0. Consider further tuning.")
    print('='*60)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(preds, truths, title, save_path):
    cond_labels = [c.split('_')[1] for c in CONDITION_NAMES]
    cm = confusion_matrix(truths, preds, labels=list(range(9)))
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(9), yticks=np.arange(9),
        xticklabels=cond_labels, yticklabels=cond_labels,
        title=title, ylabel='True label', xlabel='Predicted label',
    )
    thresh = cm.max() / 2.0
    for i in range(9):
        for j in range(9):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


def _plot_training_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history['train_loss'], label='Train loss', color='steelblue')
    ax1.plot(epochs, history['val_loss'],   label='Val loss',   color='tomato')
    ax1.set(title='Loss', xlabel='Epoch', ylabel='Loss')
    ax1.legend()

    ax2.plot(epochs, history['val_acc'], color='seagreen')
    ax2.set(title='Validation Accuracy', xlabel='Epoch', ylabel='Accuracy')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curve saved → {save_path}")


def _plot_esi_bars(esi_results, save_path):
    """Bar chart of clean ESI scores per condition."""
    names = list(esi_results.keys())
    conds = [n.split('_')[1] for n in names]
    means = [esi_results[n]['mean_esi'] for n in names]
    stds  = [esi_results[n]['std_esi']  for n in names]
    best_idx = int(np.argmax(means))
    colors = ['gold' if i == best_idx else 'steelblue' for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(conds, means, yerr=stds, color=colors, capsize=5, edgecolor='white')
    ax.set_title('ESI by Condition (clean, no manipulation)')
    ax.set_xlabel('Environmental Condition')
    ax.set_ylabel('ESI Score')
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.001, f'{m:.4f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ESI plot saved → {save_path}")


def _plot_ablation(all_results, save_path):
    names = list(all_results.keys())
    accs  = [all_results[n]['mean_acc'] for n in names]
    stds  = [all_results[n]['std_acc']  for n in names]

    labels = [n.replace('_', '\n') for n in names]
    colors = ['tomato' if n == 'full' else 'steelblue' for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, accs, yerr=stds, color=colors,
                  capsize=5, edgecolor='white')
    ax.set_title('Modality Ablation Study', fontsize=13)
    ax.set_ylabel('Accuracy (5-fold CV mean ± std)')
    ax.set_ylim(0, 1.05)
    for bar, acc, std in zip(bars, accs, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Ablation plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(phase_name, results):
    print(f"\n{'─'*50}")
    print(f"{phase_name} RESULT")
    print(f"  Acc:  {results['mean_acc']:.4f} ± {results['std_acc']:.4f}")
    print(f"  F1:   {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
    print('─'*50)


def _print_ablation_table(all_results):
    print(f"\n{'─'*55}")
    print(f"{'Configuration':<18} {'Acc (mean)':>10} {'±':>3} {'std':>6}  {'F1 (mean)':>10}")
    print('─'*55)
    for name, r in all_results.items():
        marker = ' ◀' if name == 'full' else ''
        print(f"{name:<18} {r['mean_acc']:>10.4f} {'±':>3} {r['std_acc']:>6.4f}  "
              f"{r['mean_f1']:>10.4f}{marker}")
    print('─'*55)


def _print_baseline_table(all_results):
    print(f"\n{'─'*55}")
    print(f"{'Model':<25} {'Acc (mean)':>10} {'±':>3} {'std':>6}")
    print('─'*55)
    for name, r in all_results.items():
        print(f"{name:<25} {r['mean_acc']:>10.4f} {'±':>3} {r['std_acc']:>6.4f}")
    print('─'*55)


def _print_fusion_table(all_results):
    print(f"\n{'─'*55}")
    print(f"{'Fusion Strategy':<22} {'Acc (mean)':>10} {'±':>3} {'std':>6}")
    print('─'*55)
    for name, r in all_results.items():
        marker = ' ◀ ours' if name == 'self_attention' else ''
        print(f"{name:<22} {r['mean_acc']:>10.4f} {'±':>3} {r['std_acc']:>6.4f}{marker}")
    print('─'*55)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='PRICAI 2026 Experiment Runner')
    p.add_argument('--phase',      type=int, required=True,
                   help='Experiment phase: 0=base, 1=ablation, 2=baselines, '
                        '3=fusion, 4=attention, 5=perclass')
    p.add_argument('--data_path',  type=str, required=True,
                   help='Path to experiment/data directory')
    p.add_argument('--output_dir', type=str, default='./results',
                   help='Directory to save all results')
    p.add_argument('--folds',      type=int, default=5,
                   help='Number of CV folds (default 5)')
    p.add_argument('--epochs',     type=int, default=200,
                   help='Max training epochs per fold')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr',         type=float, default=5e-4)
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Path to .pth checkpoint (required for phases 4 and 5)')
    p.add_argument('--augment', action='store_true',
                   help='Enable data augmentation during CRENet training '
                        '(flip, rotation, noise, colour jitter)')
    p.add_argument('--debug', action='store_true',
                   help='Debug mode: 2 samples/class, 2 epochs, 2 folds — '
                        'validates code without GPU training')
    return p.parse_args()


def main():
    args = parse_args()

    if args.debug:
        print("\n" + "!"*60)
        print("! DEBUG MODE — small data, 2 epochs, 2 folds")
        print("! Use this to validate code, NOT for real results")
        print("!"*60 + "\n")
        args.epochs     = 2
        args.folds      = 2
        args.batch_size = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    t0 = time.time()

    # Load data (shared across all phases)
    optical, sem, eds, labels = load_dataset(args.data_path, debug=args.debug)

    dispatch = {
        0: run_phase0,
        1: run_phase1,
        2: run_phase2,
        3: run_phase3,
        4: run_phase4,
        5: run_phase5,
        6: run_phase6,
    }

    if args.phase not in dispatch:
        print(f"Unknown phase {args.phase}. Valid phases: {list(dispatch.keys())}")
        sys.exit(1)

    results = dispatch[args.phase](args, device, optical, sem, eds, labels)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed/60:.1f} min")
    return results


if __name__ == '__main__':
    main()
