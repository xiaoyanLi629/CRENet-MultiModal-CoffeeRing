"""
Honest per-sample ESI analysis for camera-ready revision.

Computes, from raw EDS maps with NO condition-specific adjustment factors:
  1. Per-sample S_intra, S_inter, ESI (at alpha* and alpha=0.6)
  2. One-way ANOVA + Kruskal-Wallis of ESI across the 9 conditions
  3. Sensitivity: threshold percentile {60,70,80} x subsample {50,100,200}
  4. Alpha-grid ranking stability (which condition ranks 1st at each alpha)

Run from the project root:  python3 esi_honest_analysis.py
"""
import os, sys, json
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from experiments.esi_calibration import _eds_to_positions, _s_intra, _s_inter
from experiments.train_utils import load_dataset, CONDITION_NAMES

RNG_SEED = 42

def per_sample_scores(eds, labels, threshold_pct=70, max_pts_inter=100, max_pts_intra=200):
    np.random.seed(RNG_SEED)
    intra, inter = [], []
    for k in range(len(labels)):
        pos = _eds_to_positions(eds[k], threshold_pct=threshold_pct)
        si = _s_intra(pos, max_pts=max_pts_intra)
        sr = _s_inter(pos, max_pts=max_pts_inter)        # raw
        sr = sr / (1.0 + sr)                             # normalise
        intra.append(si); inter.append(sr)
    return np.array(intra), np.array(inter)

def condition_stats(vals, labels):
    out = {}
    for c in range(9):
        v = vals[labels == c]
        out[CONDITION_NAMES[c]] = {"mean": float(v.mean()), "std": float(v.std()), "n": int(len(v))}
    return out

def anova_tests(vals, labels):
    groups = [vals[labels == c] for c in range(9)]
    f, p = stats.f_oneway(*groups)
    h, pk = stats.kruskal(*groups)
    # eta squared
    grand = vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((vals - grand) ** 2).sum()
    return {"anova_F": float(f), "anova_p": float(p),
            "kruskal_H": float(h), "kruskal_p": float(pk),
            "eta_sq": float(ss_between / ss_total)}

def ranking(vals, labels):
    means = np.array([vals[labels == c].mean() for c in range(9)])
    order = np.argsort(-means)
    return [chr(65 + int(i)) for i in order]

def main():
    optical, sem, eds, labels = load_dataset("./data", debug=False)
    del optical, sem
    print(f"EDS loaded: {eds.shape}, labels: {np.bincount(labels)}")

    results = {}

    # ---- main setting (70th pct, 100/200 pts) ----
    s_intra, s_inter = per_sample_scores(eds, labels)
    var_intra = np.var([s_intra[labels == c].mean() for c in range(9)])
    var_inter = np.var([s_inter[labels == c].mean() for c in range(9)])
    alpha_star = var_intra / (var_intra + var_inter)
    esi_star = alpha_star * s_intra + (1 - alpha_star) * s_inter
    esi_06 = 0.6 * s_intra + 0.4 * s_inter

    results["alpha_star"] = float(alpha_star)
    results["esi_at_alpha_star"] = {
        "by_condition": condition_stats(esi_star, labels),
        "tests": anova_tests(esi_star, labels),
        "ranking_desc": ranking(esi_star, labels),
    }
    results["esi_at_alpha_0.6"] = {
        "by_condition": condition_stats(esi_06, labels),
        "tests": anova_tests(esi_06, labels),
        "ranking_desc": ranking(esi_06, labels),
    }
    results["s_inter_tests"] = anova_tests(s_inter, labels)
    results["s_intra_tests"] = anova_tests(s_intra, labels)

    # ---- alpha grid ranking stability ----
    grid = {}
    for a in [0.0, 0.001, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        esi_a = a * s_intra + (1 - a) * s_inter
        r = ranking(esi_a, labels)
        grid[str(a)] = {"rank1": r[0], "ranking": r}
    results["alpha_grid"] = grid

    # ---- threshold / subsample sensitivity ----
    sens = {}
    for thr in [60, 70, 80]:
        for mp in [50, 100, 200]:
            si, sr = per_sample_scores(eds, labels, threshold_pct=thr, max_pts_inter=mp)
            vi = np.var([si[labels == c].mean() for c in range(9)])
            vr = np.var([sr[labels == c].mean() for c in range(9)])
            a_s = vi / (vi + vr)
            esi = a_s * si + (1 - a_s) * sr
            key = f"thr{thr}_pts{mp}"
            sens[key] = {
                "alpha_star": float(a_s),
                "ranking_desc": ranking(esi, labels),
                "tests": anova_tests(esi, labels),
                "range": [float(min(esi[labels == c].mean() for c in range(9))),
                          float(max(esi[labels == c].mean() for c in range(9)))],
            }
            print(f"{key}: rank1={sens[key]['ranking_desc'][0]}, "
                  f"anova_p={sens[key]['tests']['anova_p']:.4g}")
    results["sensitivity"] = sens

    out = os.path.join(ROOT, "results", "esi_honest")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "esi_honest_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("Saved ->", os.path.join(out, "esi_honest_results.json"))

    # concise console summary
    print("\n===== SUMMARY =====")
    print("alpha* =", alpha_star)
    print("ranking at alpha*:", results["esi_at_alpha_star"]["ranking_desc"])
    print("ANOVA p:", results["esi_at_alpha_star"]["tests"]["anova_p"])
    print("rank1 across alpha grid:", {k: v["rank1"] for k, v in grid.items()})

if __name__ == "__main__":
    main()
