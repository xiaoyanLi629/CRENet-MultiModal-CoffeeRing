"""
experiments/models.py
=====================
All model definitions for ICANN 2026 experiments.

Models:
  - CRENet               : Original CRENet (scratch CNN encoders, self-attention fusion)
  - CRENetPretrained     : CRENet v2 — pretrained EfficientNet-B0 for visual modalities,
                           scratch CNN for EDS, late_concat fusion.
                           Motivated by Phase 2+3 findings:
                           (a) pretrained baselines significantly outperform scratch models
                           (b) late_concat outperforms self_attention on this dataset
  - EarlyFusionCNN      : All modalities concatenated at pixel level (fusion ablation)
  - ResNet18Baseline    : Single-branch ResNet-18 on optical only
  - EfficientNetBaseline: Single-branch EfficientNet-B0 on optical only
  - MultiModalResNetBaseline: Three-branch ResNet-18 for multi-modal comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Helper: shared optical / SEM encoder (5-layer CNN, unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def _make_visual_encoder(in_channels: int) -> nn.Sequential:
    """5-layer CNN encoder for 512×512 optical or SEM images → 256-dim vector."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # →256
        nn.Conv2d(32, 64, 3, padding=1),  nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),          # →128
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),          # →64
        nn.Conv2d(128, 256, 3, padding=1),nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),          # →32
        nn.Conv2d(256, 256, 3, padding=1),nn.BatchNorm2d(256), nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),   # →(256,4,4)
        nn.Flatten(),                   # →4096
        nn.Linear(4096, 512), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(512, 256),  nn.Dropout(0.3),
    )


def _make_eds_cnn_encoder(num_elements: int = 7) -> nn.Sequential:
    """
    CNN-based EDS encoder for 256×256 element maps → 128-dim vector.

    Key change vs original ml_framework.py:
      OLD: Linear(7*256*256=458752, 1024) → ~470M params (broken)
      NEW: 3-layer CNN + FC             → ~0.65M params (correct)

    Using CNN preserves the spatial structure of element distribution maps,
    which is more appropriate than flattening.
    """
    return nn.Sequential(
        nn.Conv2d(num_elements, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # →128
        nn.Conv2d(32, 64, 3, padding=1),            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), # →64
        nn.Conv2d(64, 128, 3, padding=1),           nn.BatchNorm2d(128), nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),  # →(128,4,4)
        nn.Flatten(),                  # →2048
        nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 128),  nn.Dropout(0.3),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CRENet  (used for Phase 0, 1, 3, 4, 5)
# ─────────────────────────────────────────────────────────────────────────────

FUSION_MODES = ('self_attention', 'cross_attention', 'late_concat', 'gated')


class CRENet(nn.Module):
    """
    CRENet: Coffee-Ring Effect Network (proposed model).

    A multi-modal CNN for classifying coffee-ring patterns from three
    heterogeneous imaging modalities (optical, SEM, EDS).

    Key design choices:
        - Independent 5-layer CNN encoders for optical (RGB) and SEM (grayscale)
        - CNN-based EDS encoder preserving spatial element distribution structure
          (~0.65 M params vs ~470 M in the original FC-based design)
        - Two-stage fusion: self-attention aligns visual modalities,
          chemical (EDS) features are fused independently afterwards
        - Modality mask parameter enables ablation studies without retraining

    Args:
        num_classes : number of output classes (default 9, conditions A–I)
        num_elements: number of EDS element channels (default 7)
        fusion      : one of ('self_attention', 'cross_attention',
                              'late_concat', 'gated')

    Parameter count (approx):
        optical encoder : ~3.2 M
        SEM encoder     : ~3.2 M
        EDS CNN encoder : ~0.65 M
        fusion + head   : ~0.20 M
        ─────────────────────────
        Total           : ~7.25 M
    """

    def __init__(
        self,
        num_classes: int = 9,
        num_elements: int = 7,
        fusion: str = 'self_attention',
    ):
        super().__init__()
        assert fusion in FUSION_MODES, f"fusion must be one of {FUSION_MODES}"
        self.fusion = fusion

        # ── Encoders ──────────────────────────────────────────────────────────
        self.optical_encoder = _make_visual_encoder(in_channels=3)
        self.sem_encoder     = _make_visual_encoder(in_channels=1)
        self.eds_encoder     = _make_eds_cnn_encoder(num_elements)

        # ── Fusion modules ────────────────────────────────────────────────────
        # Self-attention: Q=K=V=stack([optical, sem])  shape (B, 2, 256)
        # Note: batch_first=True fixes the original code's incorrect dimension order.
        self.self_attn = nn.MultiheadAttention(
            embed_dim=256, num_heads=4, batch_first=True, dropout=0.1
        )

        # Cross-attention: Q=optical (1 token), K=V=sem (1 token)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=256, num_heads=4, batch_first=True, dropout=0.1
        )

        # Gated fusion: learnable per-modality gate weights
        self.gate_net = nn.Sequential(
            nn.Linear(256 + 256, 2),
            nn.Softmax(dim=-1),
        )

        # ── Classification heads ──────────────────────────────────────────────
        # Attention / cross-attn / gated → attended(256) + eds(128) = 384
        self.head_384 = nn.Sequential(
            nn.Linear(384, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        # Late-concat → optical(256) + sem(256) + eds(128) = 640
        self.head_640 = nn.Sequential(
            nn.Linear(640, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        optical,
        sem,
        eds,
        modality_mask: dict = None,
        return_attention: bool = False,
    ):
        """
        Args:
            optical          : (B, 3, 512, 512)
            sem              : (B, 1, 512, 512)
            eds              : (B, 7, 256, 256)
            modality_mask    : e.g. {'optical': True, 'sem': False, 'eds': True}
                               Missing keys default to True.
            return_attention : if True, also return attention weight tensor

        Returns:
            logits (B, num_classes)  [, attn_weights if return_attention]
        """
        if modality_mask is None:
            modality_mask = {}

        # ── Encode ────────────────────────────────────────────────────────────
        opt_feat = self.optical_encoder(optical)  # (B, 256)
        sem_feat = self.sem_encoder(sem)           # (B, 256)
        eds_feat = self.eds_encoder(eds)           # (B, 128)

        # ── Apply modality mask (ablation) ────────────────────────────────────
        if not modality_mask.get('optical', True):
            opt_feat = torch.zeros_like(opt_feat)
        if not modality_mask.get('sem', True):
            sem_feat = torch.zeros_like(sem_feat)
        if not modality_mask.get('eds', True):
            eds_feat = torch.zeros_like(eds_feat)

        # ── Fuse ──────────────────────────────────────────────────────────────
        attn_weights = None

        if self.fusion == 'self_attention':
            # Stack visual tokens: (B, 2, 256)
            visual = torch.stack([opt_feat, sem_feat], dim=1)
            attended, attn_weights = self.self_attn(visual, visual, visual)
            visual_fused = attended.mean(dim=1)           # (B, 256)
            fused = torch.cat([visual_fused, eds_feat], dim=1)  # (B, 384)
            logits = self.head_384(fused)

        elif self.fusion == 'cross_attention':
            # Q = optical (1 token), K=V = sem (1 token)
            q = opt_feat.unsqueeze(1)  # (B, 1, 256)
            kv = sem_feat.unsqueeze(1) # (B, 1, 256)
            attended, attn_weights = self.cross_attn(q, kv, kv)
            visual_fused = attended.squeeze(1)            # (B, 256)
            fused = torch.cat([visual_fused, eds_feat], dim=1)
            logits = self.head_384(fused)

        elif self.fusion == 'late_concat':
            # Direct concatenation, no attention
            fused = torch.cat([opt_feat, sem_feat, eds_feat], dim=1)  # (B, 640)
            logits = self.head_640(fused)

        elif self.fusion == 'gated':
            # Learnable gate: softmax over [optical, sem] contributions
            gate_in = torch.cat([opt_feat, sem_feat], dim=1)  # (B, 512)
            gates = self.gate_net(gate_in)                     # (B, 2)
            visual_fused = (
                gates[:, 0:1] * opt_feat + gates[:, 1:2] * sem_feat
            )                                                   # (B, 256)
            fused = torch.cat([visual_fused, eds_feat], dim=1)
            logits = self.head_384(fused)

        if return_attention:
            return logits, attn_weights
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# EarlyFusionCNN  (fusion ablation: all modalities merged at pixel level)
# ─────────────────────────────────────────────────────────────────────────────

class EarlyFusionCNN(nn.Module):
    """
    Early fusion baseline: all three modalities are resized to 256×256 and
    concatenated channel-wise (3+1+7=11 channels) before a single CNN.

    This tests whether learning from raw concatenated inputs can match
    the multi-branch attention approach.
    """

    def __init__(self, num_classes: int = 9, num_elements: int = 7):
        super().__init__()
        in_ch = 3 + 1 + num_elements  # 11

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),    nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),   nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),  nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(4096, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.3),
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, optical, sem, eds, **kwargs):
        # Resize all inputs to 256×256
        optical_r = F.interpolate(optical, size=(256, 256), mode='bilinear', align_corners=False)
        sem_r     = F.interpolate(sem,     size=(256, 256), mode='bilinear', align_corners=False)
        # eds is already 256×256
        combined = torch.cat([optical_r, sem_r, eds], dim=1)  # (B, 11, 256, 256)
        feat = self.encoder(combined)
        return self.head(feat)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline models  (Phase 2)
# ─────────────────────────────────────────────────────────────────────────────

class ResNet18Baseline(nn.Module):
    """
    Single-branch ResNet-18 on optical images only.
    Tests pretrained vs from-scratch, and compares with multi-modal approach.
    """

    def __init__(self, num_classes: int = 9, pretrained: bool = True):
        super().__init__()
        try:
            import torchvision.models as tvm
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tvm.resnet18(weights=weights)
        except AttributeError:
            # Older torchvision API
            import torchvision.models as tvm
            backbone = tvm.resnet18(pretrained=pretrained)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.model = backbone

    def forward(self, optical, sem=None, eds=None, **kwargs):
        # Resize optical to 224×224 for ResNet
        x = F.interpolate(optical, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EfficientNetBaseline(nn.Module):
    """EfficientNet-B0 on optical images only."""

    def __init__(self, num_classes: int = 9, pretrained: bool = True):
        super().__init__()
        try:
            import torchvision.models as tvm
            weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tvm.efficientnet_b0(weights=weights)
        except AttributeError:
            import torchvision.models as tvm
            backbone = tvm.efficientnet_b0(pretrained=pretrained)

        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, num_classes)
        self.model = backbone

    def forward(self, optical, sem=None, eds=None, **kwargs):
        x = F.interpolate(optical, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ViTBaseline(nn.Module):
    """Vision Transformer (ViT-Small/16) on optical images only. Requires timm."""

    def __init__(self, num_classes: int = 9, pretrained: bool = True):
        super().__init__()
        try:
            import timm
            self.model = timm.create_model(
                'vit_small_patch16_224',
                pretrained=pretrained,
                num_classes=num_classes,
            )
            self._available = True
        except ImportError:
            print("[ViTBaseline] WARNING: timm not installed. ViT baseline disabled.")
            self._available = False
            self.model = None

    def forward(self, optical, sem=None, eds=None, **kwargs):
        if not self._available:
            raise RuntimeError("timm not installed — ViT baseline unavailable.")
        x = F.interpolate(optical, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model(x)

    def count_parameters(self) -> int:
        if self._available:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return 0


class MultiModalResNetBaseline(nn.Module):
    """
    Three-branch ResNet-18 baseline (heavyweight multi-modal comparison).
    Each modality has its own ResNet-18 backbone; outputs are concatenated
    and passed to a shared classification head.

    Params: ~3 × 11.2M backbone + small head ≈ 33M
    """

    def __init__(self, num_classes: int = 9, pretrained: bool = False):
        super().__init__()
        import torchvision.models as tvm

        def _make_branch(in_ch, pretrained):
            try:
                weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if (pretrained and in_ch == 3) else None
                b = tvm.resnet18(weights=weights)
            except AttributeError:
                b = tvm.resnet18(pretrained=(pretrained and in_ch == 3))
            # Adapt first conv for non-RGB inputs
            if in_ch != 3:
                b.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
            b.fc = nn.Identity()  # remove classification head; use pooled features
            return b

        self.optical_branch = _make_branch(3, pretrained)
        self.sem_branch      = _make_branch(1, False)
        self.eds_branch      = _make_branch(7, False)

        feat_dim = 512  # ResNet-18 avgpool output
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 3, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, optical, sem, eds, **kwargs):
        optical_r = F.interpolate(optical, size=(224, 224), mode='bilinear', align_corners=False)
        sem_r     = F.interpolate(sem,     size=(224, 224), mode='bilinear', align_corners=False)
        eds_r     = F.interpolate(eds,     size=(224, 224), mode='bilinear', align_corners=False)

        f_opt = self.optical_branch(optical_r)  # (B, 512)
        f_sem = self.sem_branch(sem_r)           # (B, 512)
        f_eds = self.eds_branch(eds_r)           # (B, 512)

        fused = torch.cat([f_opt, f_sem, f_eds], dim=1)  # (B, 1536)
        return self.head(fused)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# CRENetPretrained  — Improved CRENet motivated by ablation findings
# ─────────────────────────────────────────────────────────────────────────────

class CRENetPretrained(nn.Module):
    """
    CRENetPretrained: CRENet with pretrained EfficientNet-B0 visual encoders.

    Motivation (from Phase 2 & 3 experimental findings):
      • Phase 2: pretrained EfficientNet-B0 optical-only (92%) >>
                 scratch CRENet multi-modal (76.89%). Pretrained features
                 are essential with only 20 training samples per class.
      • Phase 3: late_concat (80.44%) > self_attention (74.22%) on this dataset.

    Architecture:
      optical → pretrained EfficientNet-B0            → 256-dim
      SEM     → pretrained EfficientNet-B0 (grayscale-adapted) → 256-dim
      EDS     → scratch 3-layer CNN                   → 128-dim
      fusion  → late_concat (640-dim) → FC → 9 classes

    Total parameters: ~11.4M
    """

    def __init__(self, num_classes: int = 9, num_elements: int = 7):
        super().__init__()
        import torchvision.models as tvm

        # ── Optical: pretrained EfficientNet-B0 ──────────────────────────────
        try:
            eff_opt = tvm.efficientnet_b0(
                weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        except AttributeError:
            eff_opt = tvm.efficientnet_b0(pretrained=True)
        in_feat = eff_opt.classifier[1].in_features   # 1280
        eff_opt.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_feat, 256), nn.ReLU())
        self.optical_encoder = eff_opt

        # ── SEM: pretrained EfficientNet-B0 adapted for grayscale ────────────
        try:
            eff_sem = tvm.efficientnet_b0(
                weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        except AttributeError:
            eff_sem = tvm.efficientnet_b0(pretrained=True)
        orig_conv = eff_sem.features[0][0]
        new_conv = nn.Conv2d(1, orig_conv.out_channels,
                             kernel_size=orig_conv.kernel_size,
                             stride=orig_conv.stride,
                             padding=orig_conv.padding, bias=False)
        # Initialise grayscale conv by averaging RGB channel weights
        new_conv.weight.data = orig_conv.weight.data.mean(dim=1, keepdim=True)
        eff_sem.features[0][0] = new_conv
        in_feat_sem = eff_sem.classifier[1].in_features
        eff_sem.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_feat_sem, 256), nn.ReLU())
        self.sem_encoder = eff_sem

        # ── EDS: scratch CNN (no pretrained weights available) ────────────────
        self.eds_encoder = _make_eds_cnn_encoder(num_elements)   # → 128-dim

        # ── Late-concat classifier ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(640, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, optical, sem, eds, modality_mask=None, **kwargs):
        optical_r = F.interpolate(optical, (224, 224), mode='bilinear', align_corners=False)
        sem_r     = F.interpolate(sem,     (224, 224), mode='bilinear', align_corners=False)

        opt_feat = self.optical_encoder(optical_r)   # (B, 256)
        sem_feat = self.sem_encoder(sem_r)            # (B, 256)
        eds_feat = self.eds_encoder(eds)              # (B, 128)

        if modality_mask:
            if not modality_mask.get('optical', True):
                opt_feat = torch.zeros_like(opt_feat)
            if not modality_mask.get('sem', True):
                sem_feat = torch.zeros_like(sem_feat)
            if not modality_mask.get('eds', True):
                eds_feat = torch.zeros_like(eds_feat)

        fused = torch.cat([opt_feat, sem_feat, eds_feat], dim=1)  # (B, 640)
        return self.classifier(fused)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def backbone_parameters(self):
        """Separate backbone vs head params for differential LR."""
        bb = (list(self.optical_encoder.features.parameters()) +
              list(self.sem_encoder.features.parameters()))
        head = (list(self.optical_encoder.classifier.parameters()) +
                list(self.sem_encoder.classifier.parameters()) +
                list(self.eds_encoder.parameters()) +
                list(self.classifier.parameters()))
        return bb, head


# ─────────────────────────────────────────────────────────────────────────────
# CRENetEfficient  — Proven design motivated by Phase 6 findings
# ─────────────────────────────────────────────────────────────────────────────

class CRENetEfficient(nn.Module):
    """
    CRENetEfficient: the final proposed architecture.

    Motivated by Phase 6 findings:
      - Two competing pretrained backbones (optical + adapted SEM) interfere
        during training and give worse results than scratch CRENet.
      - SEM adapted from RGB pretrained weights performs poorly due to the
        fundamental domain gap between natural images and electron micrographs.
      - Solution: use the proven pretrained EfficientNet-B0 for optical only,
        and scratch CNNs for SEM and EDS (which have no suitable pretrained
        weights). Fine-tune all parameters with a uniform LR (same setup
        that gave 92% in the EfficientNet baseline).

    Architecture:
      optical → pretrained EfficientNet-B0 (fine-tuned) → 1280-dim
      SEM     → scratch 5-layer CNN                      → 256-dim
      EDS     → scratch 3-layer CNN                      → 128-dim
      fusion  → late_concat (1664-dim) → FC → 9 classes

    This answers the key research question:
      "Does multi-modal fusion (optical + SEM + EDS) improve over the
       best single-modal pretrained baseline (optical EfficientNet-B0, 92%)?"

    Total parameters: ~8.1M
    """

    def __init__(self, num_classes: int = 9, num_elements: int = 7):
        super().__init__()
        import torchvision.models as tvm

        # ── Optical: pretrained EfficientNet-B0 (full fine-tune) ─────────────
        try:
            eff = tvm.efficientnet_b0(
                weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        except AttributeError:
            eff = tvm.efficientnet_b0(pretrained=True)
        # Keep all 1280 features from avgpool; remove original classifier
        eff.classifier = nn.Identity()
        self.optical_encoder = eff                      # → 1280-dim

        # ── SEM: scratch 5-layer CNN (no suitable pretrained weights) ─────────
        self.sem_encoder = _make_visual_encoder(in_channels=1)  # → 256-dim

        # ── EDS: scratch 3-layer CNN ──────────────────────────────────────────
        self.eds_encoder = _make_eds_cnn_encoder(num_elements)  # → 128-dim

        # ── Late-concat: 1280 + 256 + 128 = 1664 ─────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(1664, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, optical, sem, eds, modality_mask=None, **kwargs):
        # Resize optical to 224×224 (EfficientNet input size)
        optical_r = F.interpolate(optical, (224, 224),
                                  mode='bilinear', align_corners=False)

        opt_feat = self.optical_encoder(optical_r)   # (B, 1280)
        sem_feat = self.sem_encoder(sem)              # (B, 256)
        eds_feat = self.eds_encoder(eds)              # (B, 128)

        if modality_mask:
            if not modality_mask.get('optical', True):
                opt_feat = torch.zeros_like(opt_feat)
            if not modality_mask.get('sem', True):
                sem_feat = torch.zeros_like(sem_feat)
            if not modality_mask.get('eds', True):
                eds_feat = torch.zeros_like(eds_feat)

        fused = torch.cat([opt_feat, sem_feat, eds_feat], dim=1)  # (B, 1664)
        return self.classifier(fused)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
