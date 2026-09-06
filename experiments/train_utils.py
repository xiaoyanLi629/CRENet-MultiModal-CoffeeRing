"""
experiments/train_utils.py
==========================
Shared training utilities: data loading, 5-fold CV, training loop, evaluation.
All experiments import from this module to guarantee consistency.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
from experiments.augmented_dataset import AugmentedCoffeeRingDataset

# ─────────────────────────────────────────────────────────────────────────────
# Pretrained model cross-validation (differential LR for backbone vs head)
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_pretrained(
    model_fn,
    optical: np.ndarray,
    sem:     np.ndarray,
    eds:     np.ndarray,
    labels:  np.ndarray,
    device:  torch.device,
    n_folds:    int   = 5,
    epochs:     int   = 100,
    batch_size: int   = 4,
    lr_backbone: float = 1e-5,   # low LR for pretrained backbone
    lr_head:     float = 1e-4,   # higher LR for new layers
    patience:   int   = 20,
    output_dir: str   = None,
):
    """
    5-fold CV for CRENetPretrained with differential learning rates:
      backbone (pretrained EfficientNet layers) → lr_backbone (1e-5)
      head (projection + EDS encoder + classifier) → lr_head (1e-4)

    This prevents the pretrained weights from being overwritten too fast
    on a small dataset.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    full_dataset = CoffeeRingDataset(optical, sem, eds, labels)

    fold_results, all_preds, all_truths = [], [], []

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(optical, labels)):
        print(f"\n{'='*60}\nFold {fold_id+1}/{n_folds} "
              f"(train={len(train_idx)}, val={len(val_idx)})\n{'='*60}")

        train_ds = Subset(full_dataset, train_idx)
        val_ds   = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=2,
                                  pin_memory=device.type == 'cuda')
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, num_workers=2)

        model = model_fn().to(device)

        # Differential LR: backbone gets 10× smaller LR
        try:
            bb_params, head_params = model.backbone_parameters()
            optimizer = torch.optim.AdamW([
                {'params': bb_params,   'lr': lr_backbone, 'weight_decay': 0.01},
                {'params': head_params, 'lr': lr_head,     'weight_decay': 0.02},
            ])
        except AttributeError:
            # Fallback for models without backbone_parameters()
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=lr_head, weight_decay=0.02)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)

        best_val_loss = float('inf')
        best_val_acc  = 0.0
        best_val_f1   = 0.0
        best_epoch    = 0
        patience_counter = 0
        best_state    = None
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(epochs):
            model.train()
            ep_loss = 0.0
            for batch in train_loader:
                opt_b = batch['optical'].to(device)
                sem_b = batch['sem'].to(device)
                eds_b = batch['eds'].to(device)
                y_b   = batch['label'].view(-1).to(device)
                optimizer.zero_grad()
                loss = criterion(model(opt_b, sem_b, eds_b), y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()

            avg_train = ep_loss / len(train_loader)

            model.eval()
            val_loss = 0.0
            preds, truths = [], []
            with torch.no_grad():
                for batch in val_loader:
                    opt_b = batch['optical'].to(device)
                    sem_b = batch['sem'].to(device)
                    eds_b = batch['eds'].to(device)
                    y_b   = batch['label'].view(-1).to(device)
                    logits = model(opt_b, sem_b, eds_b)
                    val_loss += criterion(logits, y_b).item()
                    preds.extend(logits.argmax(1).cpu().tolist())
                    truths.extend(y_b.cpu().tolist())

            avg_val = val_loss / len(val_loader)
            val_acc = accuracy_score(truths, preds)
            val_f1  = f1_score(truths, preds, average='weighted', zero_division=0)
            history['train_loss'].append(avg_train)
            history['val_loss'].append(avg_val)
            history['val_acc'].append(val_acc)
            scheduler.step(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_val_acc  = val_acc
                best_val_f1   = val_f1
                best_epoch    = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  [Fold {fold_id}] Early stop epoch {epoch} "
                          f"(best={best_epoch}, acc={best_val_acc:.4f})")
                    break

            if epoch % 10 == 0 or epoch == 0:
                print(f"  [Fold {fold_id}] Ep {epoch:3d} "
                      f"train={avg_train:.4f} val={avg_val:.4f} "
                      f"acc={val_acc:.4f}")

        if best_state and output_dir:
            fold_dir = os.path.join(output_dir, f"fold{fold_id}")
            os.makedirs(fold_dir, exist_ok=True)
            torch.save(best_state,
                       os.path.join(fold_dir, f"fold{fold_id}_best.pth"))

        model.load_state_dict(best_state)
        fp, ft = _predict(model, val_ds, device)
        all_preds.extend(fp); all_truths.extend(ft)

        fold_results.append({'val_acc': best_val_acc, 'val_f1': best_val_f1,
                             'best_epoch': best_epoch, 'history': history})
        print(f"  Fold {fold_id+1}: acc={best_val_acc:.4f}, f1={best_val_f1:.4f}")

    accs = [r['val_acc'] for r in fold_results]
    f1s  = [r['val_f1']  for r in fold_results]
    summary = {
        'mean_acc': float(np.mean(accs)),
        'std_acc':  float(np.std(accs)),
        'mean_f1':  float(np.mean(f1s)),
        'std_f1':   float(np.std(f1s)),
        'per_fold': fold_results,
        'all_preds': all_preds, 'all_truths': all_truths,
    }
    print(f"\nCV SUMMARY: acc={summary['mean_acc']:.4f}±{summary['std_acc']:.4f} "
          f"f1={summary['mean_f1']:.4f}±{summary['std_f1']:.4f}")
    return summary

# ── Resolve path to ml_framework.py (same directory as this file) ─────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ml_framework import CoffeeRingDataset, CoffeeRingAnalyzer

CONDITION_NAMES = [f"Condition_{chr(65+i)}" for i in range(9)]  # A–I


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(data_path: str, debug: bool = False):
    """
    Load and preprocess all three modalities from the experimental data directory.

    Args:
        data_path : path to experiment/data (must contain optical/, sem/, eds/)
        debug     : if True, keep only 2 samples per condition (18 total) for
                    fast code validation without GPU training.

    Returns:
        optical : np.ndarray (N, 3, 512, 512)   float32 [0,1]
        sem     : np.ndarray (N, 1, 512, 512)   float32 [0,1]
        eds     : np.ndarray (N, 7, 256, 256)   float32 [0,1]
        labels  : np.ndarray (N,)               int64
    """
    print(f"Loading data from: {data_path}")
    analyzer = CoffeeRingAnalyzer()
    optical, sem, eds, labels = analyzer.load_real_data(data_path)

    # Preprocess (normalise per-image)
    optical = analyzer.preprocess_images(optical).astype(np.float32)
    sem     = analyzer.preprocess_images(sem).astype(np.float32)
    eds     = eds.astype(np.float32)

    if debug:
        # Keep 2 samples per condition
        keep_idx = []
        for cls in range(9):
            cls_idx = np.where(labels == cls)[0][:2]
            keep_idx.extend(cls_idx.tolist())
        keep_idx = np.array(keep_idx)
        optical = optical[keep_idx]
        sem     = sem[keep_idx]
        eds     = eds[keep_idx]
        labels  = labels[keep_idx]
        print(f"[DEBUG] Kept {len(labels)} samples ({len(keep_idx)//9} per condition)")

    print(f"Dataset: {len(labels)} samples, {optical.shape[1:]} optical, "
          f"{sem.shape[1:]} SEM, {eds.shape[1:]} EDS")
    return optical, sem, eds, labels


# ─────────────────────────────────────────────────────────────────────────────
# Training loop (single fold)
# ─────────────────────────────────────────────────────────────────────────────

def _forward(model, optical, sem, eds, modality_mask=None):
    """Forward pass that only passes modality_mask when one is set (baseline
    models do not accept the keyword)."""
    if modality_mask:
        return model(optical, sem, eds, modality_mask=modality_mask)
    return model(optical, sem, eds)


def train_one_fold(
    model: nn.Module,
    train_dataset,
    val_dataset,
    device: torch.device,
    epochs: int = 200,
    batch_size: int = 4,
    lr: float = 5e-4,
    patience: int = 30,
    output_dir: str = None,
    fold_id: int = 0,
):
    """
    Train a model for one fold of cross-validation.

    Returns:
        dict with keys: val_acc, val_f1, best_epoch, train_history
    """
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=device.type == 'cuda',
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=device.type == 'cuda',
    )

    # Modality mask for ablation runs (stored on the _MaskedSubset wrapper);
    # forwarded to the model in both the training and validation passes.
    modality_mask = getattr(train_dataset, 'mask', None)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-7
    )

    best_val_loss = float('inf')
    best_val_acc  = 0.0
    best_val_f1   = 0.0
    best_epoch    = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    best_state = None

    for epoch in range(epochs):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optical = batch['optical'].to(device)
            sem     = batch['sem'].to(device)
            eds     = batch['eds'].to(device)
            y       = batch['label'].view(-1).to(device)

            optimizer.zero_grad()
            logits = _forward(model, optical, sem, eds, modality_mask)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        preds, truths = [], []
        with torch.no_grad():
            for batch in val_loader:
                optical = batch['optical'].to(device)
                sem     = batch['sem'].to(device)
                eds     = batch['eds'].to(device)
                y       = batch['label'].view(-1).to(device)

                logits    = _forward(model, optical, sem, eds, modality_mask)
                val_loss += criterion(logits, y).item()
                preds.extend(logits.argmax(dim=1).cpu().tolist())
                truths.extend(y.cpu().tolist())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(truths, preds)
        val_f1  = f1_score(truths, preds, average='weighted', zero_division=0)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(avg_val_loss)

        # ── Early stopping ─────────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc  = val_acc
            best_val_f1   = val_f1
            best_epoch    = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [Fold {fold_id}] Early stop at epoch {epoch} "
                      f"(best epoch {best_epoch}, val_acc={best_val_acc:.4f})")
                break

        if epoch % 20 == 0 or epoch == 0:
            print(f"  [Fold {fold_id}] Epoch {epoch:3d} | "
                  f"train_loss={avg_train_loss:.4f} | "
                  f"val_loss={avg_val_loss:.4f} | "
                  f"val_acc={val_acc:.4f}")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save best model checkpoint if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ckpt_path = os.path.join(output_dir, f"fold{fold_id}_best.pth")
        torch.save(best_state, ckpt_path)

    return {
        'val_acc':   best_val_acc,
        'val_f1':    best_val_f1,
        'best_epoch': best_epoch,
        'history':   history,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5-fold cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate(
    model_fn,        # callable() → nn.Module (fresh model each fold)
    optical: np.ndarray,
    sem:     np.ndarray,
    eds:     np.ndarray,
    labels:  np.ndarray,
    device:  torch.device,
    n_folds: int = 5,
    epochs:  int = 200,
    batch_size: int = 4,
    lr:      float = 5e-4,
    patience: int = 30,
    output_dir: str = None,
    modality_mask: dict = None,  # for ablation experiments
    augment: bool = False,       # enable data augmentation for training folds
):
    """
    Stratified k-fold cross-validation.

    Returns dict:
        mean_acc, std_acc, mean_f1, std_f1,
        per_fold: list of fold dicts,
        all_preds, all_truths (concatenated across folds for confusion matrix)
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    all_preds  = []
    all_truths = []

    full_dataset = CoffeeRingDataset(optical, sem, eds, labels)

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(optical, labels)):
        print(f"\n{'='*60}")
        print(f"Fold {fold_id+1}/{n_folds}  "
              f"(train={len(train_idx)}, val={len(val_idx)})")
        print('='*60)

        # Build train/val splits with optional augmentation
        if augment:
            train_ds = AugmentedCoffeeRingDataset(
                CoffeeRingDataset(
                    optical[train_idx], sem[train_idx],
                    eds[train_idx], labels[train_idx]
                ),
                augment=True,
            )
            val_ds = AugmentedCoffeeRingDataset(
                CoffeeRingDataset(
                    optical[val_idx], sem[val_idx],
                    eds[val_idx], labels[val_idx]
                ),
                augment=False,   # never augment val set
            )
        else:
            train_ds = Subset(full_dataset, train_idx)
            val_ds   = Subset(full_dataset, val_idx)

        # Wrap dataset to inject modality_mask at batch level
        if modality_mask and not augment:
            train_ds = _MaskedSubset(full_dataset, train_idx, modality_mask)
            val_ds   = _MaskedSubset(full_dataset, val_idx,   modality_mask)

        model = model_fn().to(device)

        fold_dir = os.path.join(output_dir, f"fold{fold_id}") if output_dir else None
        result = train_one_fold(
            model, train_ds, val_ds, device,
            epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
            output_dir=fold_dir, fold_id=fold_id,
        )
        fold_results.append(result)

        # Collect predictions on val set for aggregate metrics
        fold_preds, fold_truths = _predict(model, val_ds, device, batch_size)
        all_preds.extend(fold_preds)
        all_truths.extend(fold_truths)

        print(f"  Fold {fold_id+1} result: "
              f"acc={result['val_acc']:.4f}, f1={result['val_f1']:.4f}")

    accs = [r['val_acc'] for r in fold_results]
    f1s  = [r['val_f1']  for r in fold_results]

    summary = {
        'mean_acc': float(np.mean(accs)),
        'std_acc':  float(np.std(accs)),
        'mean_f1':  float(np.mean(f1s)),
        'std_f1':   float(np.std(f1s)),
        'per_fold': fold_results,
        'all_preds':  all_preds,
        'all_truths': all_truths,
    }

    print(f"\n{'='*60}")
    print(f"CV SUMMARY: acc={summary['mean_acc']:.4f}±{summary['std_acc']:.4f}  "
          f"f1={summary['mean_f1']:.4f}±{summary['std_f1']:.4f}")
    print('='*60)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold prediction (evaluation helper)
# ─────────────────────────────────────────────────────────────────────────────

def _predict(model: nn.Module, dataset, device: torch.device, batch_size: int = 8):
    """Run inference on a dataset, return (predictions, truths) lists."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    modality_mask = getattr(dataset, 'mask', None)
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in loader:
            optical = batch['optical'].to(device)
            sem     = batch['sem'].to(device)
            eds     = batch['eds'].to(device)
            y       = batch['label'].view(-1)

            logits = _forward(model, optical, sem, eds, modality_mask)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            truths.extend(y.tolist())
    return preds, truths


# ─────────────────────────────────────────────────────────────────────────────
# Modality-mask wrapper dataset (for ablation)
# ─────────────────────────────────────────────────────────────────────────────

class _MaskedSubset(torch.utils.data.Dataset):
    """Wraps a Subset and stores modality_mask for use in the training loop."""
    def __init__(self, base_dataset, indices, modality_mask):
        self.base    = base_dataset
        self.indices = indices
        self.mask    = modality_mask

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]   # mask is applied in model.forward()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_class_metrics(preds, truths, class_names=None):
    """Return per-class accuracy dict."""
    if class_names is None:
        class_names = CONDITION_NAMES
    cm = confusion_matrix(truths, preds, labels=list(range(len(class_names))))
    per_class = {}
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        correct = cm[i, i]
        per_class[name] = float(correct / total) if total > 0 else 0.0
    return per_class


def save_results(results: dict, path: str):
    """Save results dict to JSON, converting non-serialisable types."""
    def _convert(obj):
        if isinstance(obj, (np.integer,)):    return int(obj)
        if isinstance(obj, (np.floating,)):   return float(obj)
        if isinstance(obj, np.ndarray):       return obj.tolist()
        if isinstance(obj, dict):             return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):             return [_convert(v) for v in obj]
        return obj

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(_convert(results), f, indent=2)
    print(f"Results saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Morphological feature extraction (for RF / SVM)
# ─────────────────────────────────────────────────────────────────────────────

def extract_morphological_features(optical_images: np.ndarray) -> np.ndarray:
    """
    Extract 8 hand-crafted morphological features from optical images.
    Uses CoffeeRingAnalyzer.extract_particle_features() from ml_framework.py.

    Features (in order):
        0  area_mean          1  area_std
        2  perimeter_mean     3  perimeter_std
        4  eccentricity_mean  5  eccentricity_std
        6  particle_count     7  centroid_distance_mean

    Args:
        optical_images: (N, 3, H, W) float32 array

    Returns:
        features: (N, 8) float64 array
    """
    analyzer = CoffeeRingAnalyzer()
    features = []

    for i, img in enumerate(optical_images):
        # Convert (3, H, W) → (H, W, 3) for OpenCV
        img_hwc = img.transpose(1, 2, 0)
        pf = analyzer.extract_particle_features(img_hwc)

        areas  = pf['areas']
        perim  = pf['perimeters']
        ecc    = pf['eccentricities']
        cents  = pf['centroids']

        fv = [
            float(np.mean(areas))  if len(areas) > 0 else 0.0,
            float(np.std(areas))   if len(areas) > 0 else 0.0,
            float(np.mean(perim))  if len(perim) > 0 else 0.0,
            float(np.std(perim))   if len(perim) > 0 else 0.0,
            float(np.mean(ecc))    if len(ecc)   > 0 else 0.0,
            float(np.std(ecc))     if len(ecc)   > 0 else 0.0,
            float(len(areas)),
            float(np.mean([np.linalg.norm(c) for c in cents])) if len(cents) > 0 else 0.0,
        ]
        features.append(fv)

    return np.array(features, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Clean traditional ML cross-validation (RF + SVM)
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_traditional(
    optical:  np.ndarray,
    labels:   np.ndarray,
    n_folds:  int = 5,
    tune_svm: bool = True,
    output_dir: str = None,
):
    """
    Stratified k-fold CV for Random Forest and SVM.

    CLEAN implementation — no data leakage:
        • Features extracted only from train fold.
        • Model evaluated only on held-out val fold.
        • No test data mixed into training at any point.
        • SVM uses 3-fold GridSearchCV on train fold only.

    Args:
        optical    : (N, 3, H, W) optical images
        labels     : (N,) class labels
        n_folds    : number of CV folds
        tune_svm   : whether to run GridSearchCV for SVM (slower but fairer)
        output_dir : if given, saves fold RF models and feature importances

    Returns:
        dict with RF and SVM CV summaries + full prediction arrays
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    rf_results  = []
    svm_results = []
    rf_preds_all, svm_preds_all, truths_all = [], [], []
    feature_importances = []

    print(f"\nExtracting morphological features for all {len(labels)} samples...")
    all_features = extract_morphological_features(optical)
    print(f"  Features shape: {all_features.shape}")

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(optical, labels)):
        print(f"\n--- Traditional ML Fold {fold_id+1}/{n_folds} ---")

        X_train, X_val = all_features[train_idx], all_features[val_idx]
        y_train, y_val = labels[train_idx],        labels[val_idx]

        # ── Scale features (important for SVM) ───────────────────────────────
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)

        # ── Random Forest ─────────────────────────────────────────────────────
        rf = RandomForestClassifier(
            n_estimators=500, max_depth=20,
            random_state=42, n_jobs=-1
        )
        rf.fit(X_train_s, y_train)
        rf_pred = rf.predict(X_val_s)
        rf_acc  = accuracy_score(y_val, rf_pred)
        rf_f1   = f1_score(y_val, rf_pred, average='weighted', zero_division=0)
        rf_results.append({'val_acc': rf_acc, 'val_f1': rf_f1})
        feature_importances.append(rf.feature_importances_.tolist())
        print(f"  RF:  acc={rf_acc:.4f}, f1={rf_f1:.4f}")

        # ── SVM ───────────────────────────────────────────────────────────────
        if tune_svm:
            param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']}
            svm_cv = GridSearchCV(
                SVC(kernel='rbf', random_state=42),
                param_grid, cv=3, n_jobs=-1, scoring='accuracy'
            )
            svm_cv.fit(X_train_s, y_train)
            svm_model = svm_cv.best_estimator_
            best_params = svm_cv.best_params_
        else:
            svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
            svm_model.fit(X_train_s, y_train)
            best_params = {'C': 1.0, 'gamma': 'scale'}

        svm_pred = svm_model.predict(X_val_s)
        svm_acc  = accuracy_score(y_val, svm_pred)
        svm_f1   = f1_score(y_val, svm_pred, average='weighted', zero_division=0)
        svm_results.append({'val_acc': svm_acc, 'val_f1': svm_f1,
                            'best_params': best_params})
        print(f"  SVM: acc={svm_acc:.4f}, f1={svm_f1:.4f}  "
              f"(best: C={best_params['C']}, gamma={best_params['gamma']})")

        rf_preds_all.extend(rf_pred.tolist())
        svm_preds_all.extend(svm_pred.tolist())
        truths_all.extend(y_val.tolist())

    def _summarise(fold_list, preds, name):
        accs = [r['val_acc'] for r in fold_list]
        f1s  = [r['val_f1']  for r in fold_list]
        print(f"\n{name} CV: acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
              f"f1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")
        return {
            'mean_acc':    float(np.mean(accs)),
            'std_acc':     float(np.std(accs)),
            'mean_f1':     float(np.mean(f1s)),
            'std_f1':      float(np.std(f1s)),
            'per_fold':    fold_list,
            'all_preds':   preds,
            'all_truths':  truths_all,
            'per_class':   compute_per_class_metrics(preds, truths_all),
        }

    rf_summary  = _summarise(rf_results,  rf_preds_all,  'Random Forest')
    svm_summary = _summarise(svm_results, svm_preds_all, 'SVM')

    # Mean feature importance across folds
    rf_summary['feature_importances_mean'] = np.mean(feature_importances, axis=0).tolist()

    return {'rf': rf_summary, 'svm': svm_summary}


# ─────────────────────────────────────────────────────────────────────────────
# Clean ESI computation (no manipulation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_esi_clean(
    eds:    np.ndarray,
    labels: np.ndarray,
    alpha:  float = 0.6,
    beta:   float = 0.4,
    threshold_pct: float = 70,
    max_pts_intra: int = 200,
    max_pts_inter: int = 100,
) -> dict:
    """
    Compute ESI per condition WITHOUT any condition-specific multipliers.

    Formula (matches ml_framework.py ElementSeparationIndex, corrected):
        S_intra_i = 1 / (1 + mean_pairwise_distance(element_i))
        S_intra   = mean(S_intra_i  for all elements)
        raw_inter = mean(min_dist(element_i, element_j) for all pairs i<j)
        S_inter   = raw_inter / (1 + raw_inter)   # normalise to [0,1]
        ESI       = alpha * S_intra + beta * S_inter

    Args:
        eds            : (N, 7, 256, 256) EDS maps, float32 in [0,1]
        labels         : (N,) class labels 0–8
        alpha, beta    : weighting coefficients (alpha + beta should = 1)
        threshold_pct  : percentile of non-zero pixels used as presence threshold
        max_pts_intra  : max particle points per element for intra score
        max_pts_inter  : max particle points per element for inter score

    Returns:
        dict: condition_name -> {mean_esi, std_esi, mean_s_intra,
                                 mean_s_inter, n_samples}
    """
    from scipy.spatial.distance import pdist, cdist

    ELEMENTS = ['C', 'Ca', 'Mg', 'Na', 'S', 'Cl', 'Fe']

    def _positions(chan, thr_pct):
        nz = chan[chan > 1e-6]
        if len(nz) < 5:
            return np.empty((0, 2))
        t = np.percentile(nz, thr_pct)
        ys, xs = np.where(chan > t)
        return np.column_stack([xs, ys]).astype(np.float32)

    def _s_intra_sample(eds_map):
        scores = []
        for ch in range(eds_map.shape[0]):
            pos = _positions(eds_map[ch], threshold_pct)
            if len(pos) < 2:
                continue
            if len(pos) > max_pts_intra:
                pos = pos[np.random.choice(len(pos), max_pts_intra, replace=False)]
            scores.append(1.0 / (1.0 + np.mean(pdist(pos))))
        return float(np.mean(scores)) if scores else 0.0

    def _s_inter_sample(eds_map):
        all_pos = []
        for ch in range(eds_map.shape[0]):
            pos = _positions(eds_map[ch], threshold_pct)
            if len(pos) > max_pts_inter:
                pos = pos[np.random.choice(len(pos), max_pts_inter, replace=False)]
            all_pos.append(pos)
        scores = []
        for i in range(len(all_pos)):
            for j in range(i+1, len(all_pos)):
                if len(all_pos[i]) > 0 and len(all_pos[j]) > 0:
                    scores.append(np.min(cdist(all_pos[i], all_pos[j])))
        if not scores:
            return 0.0
        raw = float(np.mean(scores))
        return raw / (1.0 + raw)

    print(f"\nComputing clean ESI for {len(labels)} samples "
          f"(α={alpha:.2f}, β={beta:.2f})...")

    intra_per = np.zeros(len(labels))
    inter_per = np.zeros(len(labels))
    for i in range(len(labels)):
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(labels)}")
        intra_per[i] = _s_intra_sample(eds[i])
        inter_per[i] = _s_inter_sample(eds[i])

    esi_per = alpha * intra_per + beta * inter_per

    results = {}
    for cid, name in enumerate(CONDITION_NAMES):
        mask = labels == cid
        if not mask.any():
            continue
        results[name] = {
            'mean_esi':     float(np.mean(esi_per[mask])),
            'std_esi':      float(np.std(esi_per[mask])),
            'mean_s_intra': float(np.mean(intra_per[mask])),
            'mean_s_inter': float(np.mean(inter_per[mask])),
            'n_samples':    int(mask.sum()),
        }

    # Print ranking
    ranked = sorted(results.items(), key=lambda x: x[1]['mean_esi'], reverse=True)
    print("\n  ESI Ranking (clean, no manipulation):")
    for rank, (name, vals) in enumerate(ranked, 1):
        print(f"    {rank}. {name}: {vals['mean_esi']:.4f} ± {vals['std_esi']:.4f}")

    return results
