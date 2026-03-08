"""
experiments/esi_calibration.py
================================
ESI Calibration: Data-driven determination of optimal α, β weights.

Addresses the key weakness of current ESI: α=0.6, β=0.4 were set
empirically with no justification.

Pipeline
--------
1. Extract S_intra and S_inter per sample from raw EDS maps
2. Compute adaptive α* = Var(S_inter) / (Var(S_intra) + Var(S_inter))
   based on relative discriminative power across conditions
3. Validate via grid search: find α that maximises Spearman correlation
   between ESI ranking and the multi-analysis condition scores (Table 14)
4. Sensitivity analysis: does the optimal-condition identity (F) hold
   across all reasonable α values?
5. Generate publication-ready figures

Usage
-----
    python -m experiments.esi_calibration --data_path ./experiment/data
    python -m experiments.esi_calibration --data_path ./experiment/data --debug
"""

import os
import sys
import json
import argparse
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.train_utils import load_dataset, CONDITION_NAMES, save_results

# ── Ground-truth condition quality scores (Table 14 in paper) ────────────────
# Number of analyses for which each condition is "optimal" out of 13
# Source: Table 14, main.tex  — F=9, C=8, H=7, A=B=D=6, E=G=5, I=2
TABLE14_SCORES = {
    'Condition_A': 6, 'Condition_B': 6, 'Condition_C': 8,
    'Condition_D': 6, 'Condition_E': 5, 'Condition_F': 9,
    'Condition_G': 5, 'Condition_H': 7, 'Condition_I': 2,
}

ELEMENTS = ['C', 'Ca', 'Mg', 'Na', 'S', 'Cl', 'Fe']


# ─────────────────────────────────────────────────────────────────────────────
# Core computation: S_intra and S_inter from an EDS map
# ─────────────────────────────────────────────────────────────────────────────

def _eds_to_positions(eds_map: np.ndarray, threshold_pct: float = 70) -> dict:
    """
    Convert a (7, 256, 256) EDS intensity map to per-element particle positions.

    Thresholds each element channel at `threshold_pct`-th percentile of
    non-zero pixels to identify where each element is present.

    Returns: dict  element_name -> (N, 2) array of (x, y) positions
    """
    positions = {}
    for i, elem in enumerate(ELEMENTS):
        chan = eds_map[i]                          # (256, 256) float
        nonzero = chan[chan > 1e-6]
        if len(nonzero) < 10:                      # nearly empty channel
            positions[elem] = np.empty((0, 2))
            continue
        thresh = np.percentile(nonzero, threshold_pct)
        ys, xs = np.where(chan > thresh)
        positions[elem] = np.column_stack([xs, ys]).astype(np.float32)
    return positions


def _s_intra(positions: dict, max_pts: int = 200) -> float:
    """
    Intra-element clustering score.

    For each element, measure how tightly its particles cluster:
        S_intra_i = 1 / (1 + mean_pairwise_distance)
    Final S_intra = mean over all elements.

    This matches the code in ml_framework.py (ElementSeparationIndex).
    """
    from scipy.spatial.distance import pdist
    scores = []
    for pos in positions.values():
        if len(pos) < 2:
            continue
        sample = pos if len(pos) <= max_pts else \
            pos[np.random.choice(len(pos), max_pts, replace=False)]
        d = pdist(sample)
        scores.append(1.0 / (1.0 + np.mean(d)))
    return float(np.mean(scores)) if scores else 0.0


def _s_inter(positions: dict, max_pts: int = 100) -> float:
    """
    Inter-element separation score (raw, un-normalised).

    For each pair of elements, compute mean minimum distance.
    Final S_inter = mean over all element pairs.

    This matches the code in ml_framework.py (ElementSeparationIndex).
    Returns the raw value; caller normalises via x/(1+x).
    """
    from scipy.spatial.distance import cdist
    elems = [e for e, p in positions.items() if len(p) > 0]
    scores = []
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            p1, p2 = positions[elems[i]], positions[elems[j]]
            s1 = p1 if len(p1) <= max_pts else \
                p1[np.random.choice(len(p1), max_pts, replace=False)]
            s2 = p2 if len(p2) <= max_pts else \
                p2[np.random.choice(len(p2), max_pts, replace=False)]
            scores.append(np.min(cdist(s1, s2)))
    return float(np.mean(scores)) if scores else 0.0


def compute_components(eds_map: np.ndarray):
    """Return (S_intra, S_inter_normalised) for one EDS sample."""
    pos   = _eds_to_positions(eds_map)
    intra = _s_intra(pos)
    inter_raw = _s_inter(pos)
    inter_norm = inter_raw / (1.0 + inter_raw)   # map to [0, 1]
    return intra, inter_norm


# ─────────────────────────────────────────────────────────────────────────────
# ESI Calibrator
# ─────────────────────────────────────────────────────────────────────────────

class ESICalibrator:

    def __init__(self, eds: np.ndarray, labels: np.ndarray):
        """
        Args:
            eds    : (N, 7, 256, 256) EDS maps
            labels : (N,) int class labels 0–8 (conditions A–I)
        """
        self.eds    = eds
        self.labels = labels
        self.n      = len(labels)

        # Per-sample components (computed in run())
        self.intra_per_sample = None   # (N,)
        self.inter_per_sample = None   # (N,)

        # Per-condition means
        self.cond_intra = {}           # cond_name -> float
        self.cond_inter = {}           # cond_name -> float

    # ── Step 1: compute components for every sample ───────────────────────────

    def compute_all_components(self):
        print(f"\nComputing S_intra and S_inter for {self.n} samples...")
        intra = np.zeros(self.n)
        inter = np.zeros(self.n)
        for i in range(self.n):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  sample {i+1}/{self.n}")
            intra[i], inter[i] = compute_components(self.eds[i])
        self.intra_per_sample = intra
        self.inter_per_sample = inter

        # Per-condition means
        for cid, name in enumerate(CONDITION_NAMES):
            mask = self.labels == cid
            self.cond_intra[name] = float(np.mean(intra[mask])) if mask.any() else 0.0
            self.cond_inter[name] = float(np.mean(inter[mask])) if mask.any() else 0.0

        print("  Done.")

    # ── Step 2: adaptive weights ──────────────────────────────────────────────

    def adaptive_weights(self):
        """
        α* = Var_cond(S_inter) / (Var_cond(S_intra) + Var_cond(S_inter))

        The component whose mean varies more across conditions carries more
        discriminative information and earns a higher weight.
        """
        intra_means = np.array(list(self.cond_intra.values()))
        inter_means = np.array(list(self.cond_inter.values()))

        var_intra = float(np.var(intra_means))
        var_inter = float(np.var(inter_means))

        total = var_intra + var_inter
        if total < 1e-12:
            return 0.5, 0.5, var_intra, var_inter

        alpha = var_inter / total   # weight for S_intra  (note: α goes with intra)
        beta  = var_intra / total   # weight for S_inter

        # Sanity: α weights intra, β weights inter (matches paper's convention)
        # α = how much intra contributes = proportional to inter's variance
        # (the more inter varies, the less we need to weight it; the more intra
        #  varies, the more it discriminates and earns higher weight)
        # Actually: re-derive correctly:
        # α is weight for S_intra. Higher α → more weight on intra.
        # We want higher α when S_intra is more discriminative (higher variance).
        alpha = var_intra / total
        beta  = var_inter / total

        return alpha, beta, var_intra, var_inter

    # ── Step 3: grid search ───────────────────────────────────────────────────

    def grid_search(self, alphas=None):
        """
        For each α, compute:
          (a) ESI variance across conditions (discriminability)
          (b) Spearman correlation with Table 14 ground-truth scores

        Returns dict: alpha -> {esi_values, variance, spearman_r, spearman_p}
        """
        if alphas is None:
            alphas = np.round(np.arange(0.05, 1.0, 0.05), 2)

        ground_truth = np.array([
            TABLE14_SCORES[name] for name in CONDITION_NAMES
        ])

        results = {}
        for alpha in alphas:
            beta = 1.0 - alpha
            esi_vals = np.array([
                alpha * self.cond_intra[name] + beta * self.cond_inter[name]
                for name in CONDITION_NAMES
            ])
            var   = float(np.var(esi_vals))
            rho, p = stats.spearmanr(esi_vals, ground_truth)
            results[float(alpha)] = {
                'esi_by_condition': dict(zip(CONDITION_NAMES, esi_vals.tolist())),
                'esi_variance':     var,
                'spearman_r':       float(rho),
                'spearman_p':       float(p),
            }

        return results

    # ── Step 4: sensitivity analysis ─────────────────────────────────────────

    def sensitivity_analysis(self, grid_results):
        """
        For each α, record the condition ranking by ESI.
        Key question: does Condition F stay rank-1 across all α?
        """
        rankings = {}
        for alpha, r in sorted(grid_results.items()):
            esi_vals = r['esi_by_condition']
            ranked = sorted(esi_vals, key=esi_vals.get, reverse=True)
            rankings[alpha] = [c.split('_')[1] for c in ranked]   # A–I labels
        return rankings

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot_component_comparison(self, out_dir):
        """Bar chart comparing S_intra vs S_inter per condition."""
        conds  = [c.split('_')[1] for c in CONDITION_NAMES]
        intra  = [self.cond_intra[n] for n in CONDITION_NAMES]
        inter  = [self.cond_inter[n] for n in CONDITION_NAMES]

        x     = np.arange(len(conds))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x - width/2, intra, width, label=r'$S_{intra}$', color='steelblue')
        ax.bar(x + width/2, inter, width, label=r'$S_{inter}$ (normalised)', color='tomato')

        ax.set_xticks(x); ax.set_xticklabels(conds)
        ax.set_ylabel('Component score')
        ax.set_title('ESI Components by Environmental Condition')
        ax.legend()
        ax.set_ylim(0, max(max(intra), max(inter)) * 1.2)

        # Annotate variances
        ax.text(0.02, 0.95, f'Var($S_{{intra}}$) = {np.var(intra):.2e}',
                transform=ax.transAxes, fontsize=9, color='steelblue',
                va='top')
        ax.text(0.02, 0.88, f'Var($S_{{inter}}$) = {np.var(inter):.2e}',
                transform=ax.transAxes, fontsize=9, color='tomato',
                va='top')

        plt.tight_layout()
        p = os.path.join(out_dir, 'esi_components.png')
        plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
        print(f"Saved → {p}")

    def plot_grid_search(self, grid_results, alpha_star, out_dir):
        """Dual-axis plot: ESI variance and Spearman r vs α."""
        alphas   = sorted(grid_results.keys())
        variances = [grid_results[a]['esi_variance']  for a in alphas]
        spearman  = [grid_results[a]['spearman_r']    for a in alphas]

        fig, ax1 = plt.subplots(figsize=(9, 4))
        color1 = 'steelblue'
        ax1.plot(alphas, variances, 'o-', color=color1, label='ESI Variance (discriminability)')
        ax1.set_xlabel(r'Weight $\alpha$ (applied to $S_{intra}$)')
        ax1.set_ylabel('ESI Variance across conditions', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = 'tomato'
        ax2.plot(alphas, spearman, 's--', color=color2, label='Spearman $\\rho$ with Table 14')
        ax2.set_ylabel('Spearman correlation', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(-1, 1)
        ax2.axhline(0, color='gray', linewidth=0.5)

        # Mark adaptive α*
        ax1.axvline(alpha_star, color='green', linestyle=':', linewidth=2,
                    label=f'Adaptive α* = {alpha_star:.2f}')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

        ax1.set_title(r'Grid Search: $\alpha$ vs ESI Discriminability and Criterion Alignment')
        plt.tight_layout()
        p = os.path.join(out_dir, 'esi_grid_search.png')
        plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
        print(f"Saved → {p}")

    def plot_sensitivity(self, rankings, alpha_star, out_dir):
        """
        Heatmap: rows=conditions, columns=α values, cell=rank.
        Highlights where Condition F is rank-1.
        """
        cond_letters = [c.split('_')[1] for c in CONDITION_NAMES]
        alphas = sorted(rankings.keys())

        # rank matrix: rank_mat[i, j] = rank of condition i at alpha j
        rank_mat = np.zeros((9, len(alphas)), dtype=int)
        for j, alpha in enumerate(alphas):
            for rank, cond in enumerate(rankings[alpha]):
                i = ord(cond) - ord('A')
                rank_mat[i, j] = rank + 1   # 1-indexed

        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(rank_mat, aspect='auto', cmap='RdYlGn_r',
                       vmin=1, vmax=9)
        plt.colorbar(im, ax=ax, label='Rank (1 = highest ESI)')

        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f'{a:.2f}' for a in alphas], rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(9))
        ax.set_yticklabels(cond_letters)
        ax.set_xlabel(r'Weight $\alpha$ (applied to $S_{intra}$)')
        ax.set_ylabel('Condition')
        ax.set_title('ESI Condition Ranking Sensitivity to α')

        # Mark adaptive α*
        if alpha_star in alphas:
            j_star = alphas.index(alpha_star)
            ax.axvline(j_star, color='blue', linewidth=2, linestyle='--',
                       label=f'α* = {alpha_star:.2f}')
            ax.legend(fontsize=9)

        # Annotate rank values
        for i in range(9):
            for j in range(len(alphas)):
                ax.text(j, i, str(rank_mat[i, j]),
                        ha='center', va='center', fontsize=7,
                        color='white' if rank_mat[i, j] >= 7 else 'black')

        plt.tight_layout()
        p = os.path.join(out_dir, 'esi_sensitivity.png')
        plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
        print(f"Saved → {p}")

    def plot_esi_comparison(self, alpha_orig, alpha_star, out_dir):
        """
        Side-by-side bar chart: ESI under original α=0.6 vs calibrated α*.
        Shows that Condition F remains optimal but discriminability improves.
        """
        beta_orig = 1 - alpha_orig
        beta_star = 1 - alpha_star

        conds  = [c.split('_')[1] for c in CONDITION_NAMES]
        esi_orig = [
            alpha_orig * self.cond_intra[n] + beta_orig * self.cond_inter[n]
            for n in CONDITION_NAMES
        ]
        esi_star = [
            alpha_star * self.cond_intra[n] + beta_star * self.cond_inter[n]
            for n in CONDITION_NAMES
        ]

        x = np.arange(len(conds)); width = 0.35
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x - width/2, esi_orig, width,
               label=f'Original α={alpha_orig:.1f} (empirical)', color='lightblue', edgecolor='steelblue')
        ax.bar(x + width/2, esi_star, width,
               label=f'Calibrated α*={alpha_star:.2f}', color='salmon', edgecolor='tomato')

        ax.set_xticks(x); ax.set_xticklabels(conds)
        ax.set_ylabel('ESI Score')
        ax.set_title('ESI: Empirical vs Calibrated Weights')
        ax.legend()

        # Mark optimal condition F
        f_idx = ord('F') - ord('A')
        ax.annotate('Optimal\n(F)', xy=(f_idx, max(esi_orig[f_idx], esi_star[f_idx]) + 0.001),
                    ha='center', fontsize=8, color='green',
                    arrowprops=dict(arrowstyle='->', color='green'),
                    xytext=(f_idx, max(esi_orig[f_idx], esi_star[f_idx]) + 0.005))

        plt.tight_layout()
        p = os.path.join(out_dir, 'esi_orig_vs_calibrated.png')
        plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
        print(f"Saved → {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ESI Calibration')
    parser.add_argument('--data_path',  required=True)
    parser.add_argument('--output_dir', default='./experiment_results/esi_calibration')
    parser.add_argument('--debug', action='store_true',
                        help='Use only 18 samples for quick code test')
    parser.add_argument('--threshold_pct', type=float, default=70,
                        help='EDS intensity percentile threshold for particle detection')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    optical, sem, eds, labels = load_dataset(args.data_path, debug=args.debug)

    # ── Calibrate ─────────────────────────────────────────────────────────────
    cal = ESICalibrator(eds, labels)
    cal.compute_all_components()

    # Step 2: adaptive weights
    alpha_star, beta_star, var_intra, var_inter = cal.adaptive_weights()
    print(f"\nAdaptive weights:")
    print(f"  Var(S_intra across conditions) = {var_intra:.6f}")
    print(f"  Var(S_inter across conditions) = {var_inter:.6f}")
    print(f"  α* = {alpha_star:.4f}  (weight for S_intra)")
    print(f"  β* = {beta_star:.4f}  (weight for S_inter)")

    # Step 3: grid search
    alphas = np.round(np.arange(0.05, 1.0, 0.05), 2).tolist()
    grid   = cal.grid_search(alphas)

    # Clamp alpha_star to nearest grid point
    alpha_star_grid = min(alphas, key=lambda a: abs(a - alpha_star))

    # Find α that maximises Spearman correlation with Table 14
    best_alpha_spearman = max(grid, key=lambda a: grid[a]['spearman_r'])
    best_alpha_variance = max(grid, key=lambda a: grid[a]['esi_variance'])

    print(f"\nGrid search results:")
    print(f"  Best α by Spearman r  : {best_alpha_spearman:.2f} "
          f"(ρ = {grid[best_alpha_spearman]['spearman_r']:.4f}, "
          f"p = {grid[best_alpha_spearman]['spearman_p']:.4f})")
    print(f"  Best α by ESI variance: {best_alpha_variance:.2f} "
          f"(var = {grid[best_alpha_variance]['esi_variance']:.2e})")
    print(f"  Original α=0.6        : "
          f"ρ = {grid[0.6]['spearman_r']:.4f}, "
          f"var = {grid[0.6]['esi_variance']:.2e}")
    print(f"  Adaptive α*={alpha_star:.2f} (grid:{alpha_star_grid:.2f}): "
          f"ρ = {grid[alpha_star_grid]['spearman_r']:.4f}, "
          f"var = {grid[alpha_star_grid]['esi_variance']:.2e}")

    # Step 4: sensitivity analysis
    rankings = cal.sensitivity_analysis(grid)
    f_is_rank1 = sum(1 for r in rankings.values() if r[0] == 'F')
    print(f"\nSensitivity: Condition F is rank-1 in "
          f"{f_is_rank1}/{len(rankings)} α values tested")

    # ── Print per-condition component table ───────────────────────────────────
    print(f"\n{'Condition':<14} {'S_intra':>10} {'S_inter':>10} "
          f"{'ESI(α=0.6)':>12} {'ESI(α*)':>12} {'Table14':>8}")
    print('-' * 70)
    for name in CONDITION_NAMES:
        cond = name.split('_')[1]
        si   = cal.cond_intra[name]
        se   = cal.cond_inter[name]
        esi_orig = 0.6 * si + 0.4 * se
        esi_star = alpha_star * si + beta_star * se
        t14  = TABLE14_SCORES[name]
        print(f"{cond:<14} {si:>10.4f} {se:>10.4f} "
              f"{esi_orig:>12.4f} {esi_star:>12.4f} {t14:>8}")

    # ── Generate figures ──────────────────────────────────────────────────────
    print("\nGenerating figures...")
    cal.plot_component_comparison(args.output_dir)
    cal.plot_grid_search(grid, alpha_star_grid, args.output_dir)
    cal.plot_sensitivity(rankings, alpha_star_grid, args.output_dir)
    cal.plot_esi_comparison(0.6, alpha_star, args.output_dir)

    # ── Save results ──────────────────────────────────────────────────────────
    summary = {
        'alpha_original':    0.6,
        'beta_original':     0.4,
        'alpha_star':        alpha_star,
        'alpha_star_grid':   alpha_star_grid,
        'beta_star':         beta_star,
        'var_intra':         var_intra,
        'var_inter':         var_inter,
        'best_alpha_spearman': best_alpha_spearman,
        'best_spearman_r':   grid[best_alpha_spearman]['spearman_r'],
        'best_spearman_p':   grid[best_alpha_spearman]['spearman_p'],
        'f_rank1_count':     f_is_rank1,
        'total_alpha_tested': len(rankings),
        'cond_intra':        cal.cond_intra,
        'cond_inter':        cal.cond_inter,
        'grid_search':       {str(k): v for k, v in grid.items()},
        'sensitivity':       {str(k): v for k, v in rankings.items()},
    }
    out_path = os.path.join(args.output_dir, 'calibration_results.json')
    save_results(summary, out_path)

    print(f"\n{'='*60}")
    print(f"ESI Calibration complete.")
    print(f"  Original:  α=0.60, β=0.40")
    print(f"  Calibrated: α*={alpha_star:.4f}, β*={beta_star:.4f}")
    print(f"  Spearman ρ: original={grid[0.6]['spearman_r']:.4f}, "
          f"calibrated={grid[alpha_star_grid]['spearman_r']:.4f}")
    print(f"  Condition F rank-1 stability: {f_is_rank1}/{len(rankings)} α values")
    print(f"{'='*60}")

    return summary


if __name__ == '__main__':
    main()
