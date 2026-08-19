"""Regenerate paper Fig. 'combined_results.pdf' with honest ESI data.

Panels:
  (a) ESI by condition (honest per-sample means +- std, ANOVA n.s. annotated)
  (b) classification accuracy (5-fold CV)
  (c) weighted F1 (5-fold CV)
  (d) Random-Forest feature importance
  (e) CRENetEfficient confusion matrix (concatenated fold predictions)

Run from project root. Output: paper_figures/combined_results.pdf
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "paper_figures")
os.makedirs(OUT, exist_ok=True)

# ---------- data ----------
esi = json.load(open(os.path.join(ROOT, "results/esi_honest/esi_honest_results.json")))
cond = [chr(65 + i) for i in range(9)]
bycond = esi["esi_at_alpha_star"]["by_condition"]
esi_mean = [bycond[f"Condition_{c}"]["mean"] for c in cond]
esi_std = [bycond[f"Condition_{c}"]["std"] for c in cond]
anova_p = esi["esi_at_alpha_star"]["tests"]["anova_p"]

p6 = json.load(open(os.path.join(ROOT, "results/phase6/results.json")))
preds, truths = p6["all_preds"], p6["all_truths"]

models = ["CRENet\nEfficient", "EfficientNet\nB0$^\\dagger$", "ResNet-18$^\\dagger$",
          "Random\nForest", "SVM"]
accs = [88.9, 92.0, 88.9, 65.3, 62.2]
acc_err = [5.8, 2.3, 6.1, 4.1, 5.8]
f1s = [0.888, 0.917, 0.884, 0.646, 0.608]
f1_err = [0.058, 0.023, 0.061, 0.041, 0.058]
mcolors = ["#2d7d46", "#5cb85c", "#a3d9a5", "#e8944a", "#c9433a"]

feat_names = ["Area Mean", "Particle Count", "Area Std", "Ecc. Mean",
              "Perim. Mean", "Centroid Dist.", "Perim. Std", "Ecc. Std"]
feat_imp = [0.158, 0.140, 0.138, 0.128, 0.122, 0.111, 0.109, 0.093]

# ---------- figure ----------
fig = plt.figure(figsize=(12.5, 7.2))
gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.52, wspace=1.15,
                       height_ratios=[1.0, 1.15])

# (a) ESI
ax = fig.add_subplot(gs[0, :4])
colors = ["#d98c8c"] * 9
colors[5] = "#e0a83c"
ax.bar(cond, esi_mean, yerr=esi_std, color=colors, capsize=3,
       edgecolor="white", linewidth=0.6, error_kw=dict(lw=1))
ax.set_ylim(0.40, 0.60)
ax.set_ylabel("ESI Score ($\\alpha^*$)")
ax.set_xlabel("Environmental Condition")
for i, (m, s) in enumerate(zip(esi_mean, esi_std)):
    ax.text(i, m + s + 0.006, f"{m:.3f}", ha="center", fontsize=7.5)
ax.text(0.02, 0.94,
        f"one-way ANOVA: $F=0.40$, $p={anova_p:.2f}$ (n.s.)",
        transform=ax.transAxes, fontsize=9, va="top")
ax.set_title("(a) ESI by condition (differences not significant)",
             loc="left", fontsize=10, fontweight="bold")
ax.grid(axis="y", alpha=0.25, linestyle="--")

# (b) accuracy
ax = fig.add_subplot(gs[0, 4:])
ax.bar(range(5), accs, yerr=acc_err, color=mcolors, capsize=3,
       error_kw=dict(lw=1))
ax.set_xticks(range(5))
ax.set_xticklabels(models, fontsize=6.5)
ax.set_ylim(40, 105)
ax.set_ylabel("Accuracy (%)")
for i, (m, e) in enumerate(zip(accs, acc_err)):
    ax.text(i, m + e + 1.2, f"{m:.1f}", ha="center", fontsize=7.5,
            fontweight="bold")
ax.axvline(2.5, color="gray", linestyle="--", lw=0.8)
ax.set_title("(b) Accuracy (5-fold CV)", loc="left", fontsize=10,
             fontweight="bold")
ax.grid(axis="y", alpha=0.25, linestyle="--")

# (c) F1
ax = fig.add_subplot(gs[1, :2])
ax.bar(range(5), f1s, yerr=f1_err, color=mcolors, capsize=3,
       error_kw=dict(lw=1))
ax.set_xticks(range(5))
ax.set_xticklabels(models, fontsize=6.5)
ax.set_ylim(0.4, 1.05)
ax.set_ylabel("Weighted F1")
for i, (m, e) in enumerate(zip(f1s, f1_err)):
    ax.text(i, m + e + 0.012, f"{m:.3f}", ha="center", fontsize=7)
ax.axvline(2.5, color="gray", linestyle="--", lw=0.8)
ax.set_title("(c) Weighted F1 (5-fold CV)", loc="left", fontsize=10,
             fontweight="bold")
ax.grid(axis="y", alpha=0.25, linestyle="--")

# (d) RF importance
ax = fig.add_subplot(gs[1, 2:4])
y = np.arange(len(feat_names))[::-1]
ax.barh(y, feat_imp, color=plt.cm.Blues(np.linspace(0.85, 0.35, len(feat_imp))))
ax.set_yticks(y)
ax.set_yticklabels(feat_names, fontsize=7.5)
ax.set_xlabel("Importance")
for yi, v in zip(y, feat_imp):
    ax.text(v + 0.003, yi, f"{v:.3f}", va="center", fontsize=7)
ax.set_xlim(0, 0.20)
ax.set_title("(d) RF feature importance", loc="left", fontsize=10,
             fontweight="bold")

# (e) confusion matrix
ax = fig.add_subplot(gs[1, 4:])
cm = np.zeros((9, 9), dtype=int)
for t, p in zip(truths, preds):
    cm[t, p] += 1
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=25)
ax.set_xticks(range(9)); ax.set_xticklabels(cond, fontsize=7.5)
ax.set_yticks(range(9)); ax.set_yticklabels(cond, fontsize=7.5)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
for i in range(9):
    for j in range(9):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6.5,
                    color="white" if cm[i, j] > 13 else "black")
ax.set_title("(e) CRENetEfficient confusion matrix", loc="left", fontsize=10,
             fontweight="bold")

fig.savefig(os.path.join(OUT, "combined_results.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "combined_results.png"), dpi=160,
            bbox_inches="tight")
print("saved", os.path.join(OUT, "combined_results.pdf"))
