"""
Camera-ready supplementary experiments (PRICAI 2026 revision).

Exp 1: Leave-one-recipe-out (LORO) CV for CRENetEfficient
       - folds grouped by water recipe (files 1-5 = recipe 1, ..., 21-25 = recipe 5)
       - answers R2/R5: generalisation to unseen water compositions
Exp 2: LORO CV for pretrained single-modal EfficientNet-B0 (comparison)
Exp 3: Early fusion with channel-inflated pretrained EfficientNet-B0
       - 11-channel input (3 optical + 1 SEM + 7 EDS), standard stratified 5-fold
       - answers R3/D3: does early fusion win once pretraining is available?

Protocol identical to phase2/phase6: epochs=200, batch=4, lr=5e-4,
AdamW wd=0.02, label smoothing 0.1, ReduceLROnPlateau, early stop patience=30.

Run from project root:  python camera_ready_experiments.py
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from experiments.train_utils import load_dataset, train_one_fold
from experiments.models import CRENetEfficient, EfficientNetBaseline
from ml_framework import CoffeeRingDataset
from sklearn.metrics import accuracy_score, f1_score

OUT = os.path.join(ROOT, "results", "camera_ready")
os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyFusionPretrained(nn.Module):
    """EfficientNet-B0 with the stem conv inflated from 3 to 11 input channels.

    RGB filters keep their ImageNet weights; the 8 extra channels are
    initialised with the mean RGB filter, following standard channel
    inflation practice.
    """

    def __init__(self, num_classes: int = 9):
        super().__init__()
        import torchvision.models as tvm
        eff = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        old = eff.features[0][0]                     # Conv2d(3, 32, 3, stride 2)
        new = nn.Conv2d(11, old.out_channels, kernel_size=3, stride=2,
                        padding=1, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True).repeat(1, 8, 1, 1)
        eff.features[0][0] = new
        in_features = eff.classifier[1].in_features
        eff.classifier[1] = nn.Linear(in_features, num_classes)
        self.model = eff

    def forward(self, optical, sem, eds, **kwargs):
        o = F.interpolate(optical, (224, 224), mode="bilinear", align_corners=False)
        s = F.interpolate(sem, (224, 224), mode="bilinear", align_corners=False)
        e = F.interpolate(eds, (224, 224), mode="bilinear", align_corners=False)
        return self.model(torch.cat([o, s, e], dim=1))


def recipe_labels(labels: np.ndarray) -> np.ndarray:
    """Recipe id 0-4 per sample. Loading order is condition-major with image
    numbers sorted 1..25; files 1-5 = recipe 1, ..., 21-25 = recipe 5."""
    expected = np.repeat(np.arange(9), 25)
    assert np.array_equal(labels, expected), "unexpected sample ordering"
    return np.tile(np.repeat(np.arange(5), 5), 9)


def grouped_cv(model_fn, optical, sem, eds, labels, groups, tag):
    """CV where each fold holds out one group id entirely."""
    fold_results, all_preds, all_truths = [], [], []
    for fold_id, g in enumerate(sorted(set(groups.tolist()))):
        val_idx = np.where(groups == g)[0]
        train_idx = np.where(groups != g)[0]
        print(f"\n===== {tag} fold {fold_id} (held-out group {g}): "
              f"train={len(train_idx)} val={len(val_idx)} =====", flush=True)
        train_ds = CoffeeRingDataset(optical[train_idx], sem[train_idx],
                                     eds[train_idx], labels[train_idx])
        val_ds = CoffeeRingDataset(optical[val_idx], sem[val_idx],
                                   eds[val_idx], labels[val_idx])
        model = model_fn().to(DEVICE)
        res = train_one_fold(model, train_ds, val_ds, DEVICE,
                             epochs=200, batch_size=4, lr=5e-4, patience=30,
                             output_dir=os.path.join(OUT, tag), fold_id=fold_id)
        # final predictions with best weights
        model.eval()
        preds = []
        with torch.no_grad():
            for k in range(0, len(val_idx), 8):
                sl = slice(k, min(k + 8, len(val_idx)))
                batch_idx = val_idx[sl]
                o = torch.tensor(optical[batch_idx]).float().to(DEVICE)
                s = torch.tensor(sem[batch_idx]).float().to(DEVICE)
                e = torch.tensor(eds[batch_idx]).float().to(DEVICE)
                preds.extend(model(o, s, e).argmax(1).cpu().tolist())
        all_preds.extend(preds)
        all_truths.extend(labels[val_idx].tolist())
        res_slim = {k: v for k, v in res.items() if k != "history"}
        fold_results.append(res_slim)
        print(f"  -> fold {fold_id}: acc={res['val_acc']:.4f} f1={res['val_f1']:.4f}",
              flush=True)
        del model
        torch.cuda.empty_cache()

    accs = [r["val_acc"] for r in fold_results]
    f1s = [r["val_f1"] for r in fold_results]
    summary = {
        "mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs)),
        "mean_f1": float(np.mean(f1s)), "std_f1": float(np.std(f1s)),
        "per_fold": fold_results,
        "concat_acc": float(accuracy_score(all_truths, all_preds)),
        "concat_f1": float(f1_score(all_truths, all_preds, average="weighted")),
    }
    with open(os.path.join(OUT, f"{tag}_results.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\n##### {tag}: acc={summary['mean_acc']:.4f}±{summary['std_acc']:.4f} "
          f"f1={summary['mean_f1']:.4f} #####\n", flush=True)
    return summary


def stratified_cv(model_fn, optical, sem, eds, labels, tag):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    groups = np.full(len(labels), -1)
    for fid, (_, val_idx) in enumerate(skf.split(optical, labels)):
        groups[val_idx] = fid
    return grouped_cv(model_fn, optical, sem, eds, labels, groups, tag)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    optical, sem, eds, labels = load_dataset("./data", debug=False)
    groups = recipe_labels(labels)

    results = {}
    results["loro_crenet_efficient"] = grouped_cv(
        lambda: CRENetEfficient(num_classes=9),
        optical, sem, eds, labels, groups, "loro_crenet_efficient")
    results["loro_efficientnet_b0"] = grouped_cv(
        lambda: EfficientNetBaseline(num_classes=9, pretrained=True),
        optical, sem, eds, labels, groups, "loro_efficientnet_b0")
    results["earlyfusion_pretrained_5fold"] = stratified_cv(
        lambda: EarlyFusionPretrained(num_classes=9),
        optical, sem, eds, labels, "earlyfusion_pretrained_5fold")

    with open(os.path.join(OUT, "all_results.json"), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "per_fold"}
                   for k, v in results.items()}, f, indent=1)
    print("ALL DONE")


if __name__ == "__main__":
    main()
