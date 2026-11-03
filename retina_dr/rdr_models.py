#!/usr/bin/env python
"""rdr_models.py -- the two deep retina views.

Both heads are CUMULATIVE ORDINAL over K classes: a per-patch head, a logit-MEAN pool over
the bag, and an ordinal-to-proba path that enforces monotonicity by running min before
differencing.

  PatchBagNet   the anchor. ResNet18, K random 224 px patches inside the fundus at train,
                a fixed tile grid at eval, per-patch logits, logit-mean pool.
  WholeImageNet the contrasting view. timm ResNet50 at 512 px, one forward, same head.

Grades are ICDR 0..4, so the head emits 4 cumulative logits [P(>=1), P(>=2), P(>=3), P(>=4)]
and the scored endpoints (referable >= 2, vision-threatening >= 3) are columns 1 and 2 of
that vector, read through `rdr_common.cut_scores` off the 5-class proba.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def ordinal_to_proba(cum, n_classes=5):
    """Cumulative sigmoids [...,K-1] -> K-class proba. Monotone by running min, then differenced.

    cum[:, j] is P(y >= j+1), so class 0
    mass is 1 - cum[0], class j mass is cum[j-1] - cum[j], and class K-1 mass is cum[K-2].
    """
    cum = np.atleast_2d(np.asarray(cum, np.float64))
    if cum.shape[1] != n_classes - 1:
        raise ValueError(f'expected {n_classes - 1} cumulative columns, got {cum.shape[1]}')
    cum = np.clip(np.minimum.accumulate(cum, axis=1), 0.0, 1.0)      # non-increasing
    cols = [1.0 - cum[:, 0]]
    cols += [cum[:, j - 1] - cum[:, j] for j in range(1, n_classes - 1)]
    cols.append(cum[:, n_classes - 2])
    proba = np.clip(np.stack(cols, axis=1), 0.0, None)
    return proba / proba.sum(1, keepdims=True).clip(min=1e-12)


def ordinal_targets(labels, n_classes=5):
    """y -> cumulative binary targets [N, K-1], t[:, j] = 1 if y >= j+1."""
    labels = torch.as_tensor(labels).long().view(-1, 1)
    thr = torch.arange(1, n_classes, device=labels.device).view(1, -1)
    return (labels >= thr).float()


def _backbone(name, pretrained):
    """ResNet trunk with the classifier stripped; returns (module, n_features)."""
    import timm
    m = timm.create_model(name, pretrained=pretrained, num_classes=0)
    return m, m.num_features


class PatchBagNet(nn.Module):
    """Per-patch logits, logit-mean pooled over the bag."""

    def __init__(self, n_classes=5, backbone='resnet18', pretrained=True, dropout=0.2):
        super().__init__()
        self.n_classes = n_classes
        self.trunk, n_feat = _backbone(backbone, pretrained)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_feat, n_classes - 1))

    def forward(self, bag):
        """bag: [B, K, 3, P, P] -> case cumulative logits [B, K-1]."""
        b, k = bag.shape[:2]
        feats = self.trunk(bag.reshape(b * k, *bag.shape[2:]))      # [B*K, F]
        patch_logits = self.head(feats)                             # PER-PATCH logits
        return patch_logits.reshape(b, k, self.n_classes - 1).mean(dim=1)   # logit-mean pool


class WholeImageNet(nn.Module):
    """One forward over the whole resized image; same ordinal head."""

    def __init__(self, n_classes=5, backbone='resnet50', pretrained=True, dropout=0.2):
        super().__init__()
        self.n_classes = n_classes
        self.trunk, n_feat = _backbone(backbone, pretrained)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_feat, n_classes - 1))

    def forward(self, x):
        """x: [B, 3, H, W] -> cumulative logits [B, K-1]."""
        return self.head(self.trunk(x))


# Further whole-image candidates. Each entry is a timm backbone with ImageNet-scale pretraining
# and no gated weights.
EXTRA_VIEWS = {
    'convnext': 'convnext_tiny',
    'dinov2': 'vit_small_patch14_dinov2.lvd142m',
    'effnet': 'efficientnet_b0',
}


def build(member, n_classes=5, pretrained=True, backbone=None):
    if member == 'patchbag':
        return PatchBagNet(n_classes=n_classes, pretrained=pretrained,
                           backbone=backbone or 'resnet18')
    if member == 'whole':
        return WholeImageNet(n_classes=n_classes, pretrained=pretrained,
                             backbone=backbone or 'resnet50')
    if member in EXTRA_VIEWS:
        return WholeImageNet(n_classes=n_classes, pretrained=pretrained,
                             backbone=backbone or EXTRA_VIEWS[member])
    raise ValueError(f'unknown member {member}')
