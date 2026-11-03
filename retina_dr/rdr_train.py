"""Training of the anchor, the deep and handcrafted candidates and the lesion detector, and the
recipe search on DDR valid.

Subcommands: deep, feat, lesion, tune.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
import joblib


# ==========================================================================================
# rdr_train_deep.py
# ==========================================================================================
# rdr_train_deep.py -- train one deep retina member on DDR, score it on every site.
#
# Data discipline, enforced here rather than trusted:
#   * Train on the DDR train split only. Epochs fixed up front, LAST checkpoint taken. DDR
#     valid is the only selection set: it scores each epoch (printed, divergence abort) and is
#     the objective of the recipe search (`tune`); it never selects a checkpoint within a run.
#   * DDR test and the OOD sites (APTOS, Messidor-2, `--extra-site`) never touch training,
#     checkpointing, preprocessing parameters or thresholds. They are scored once, at the end;
#     under `--blind` the scores go to disk only and nothing evaluative is printed.
#   * Endpoints referable (>= 2) and vision-threatening (>= 3) at a fixed 0.5 cut, scored
#     through `rdr_metrics.cut_scores`.
#
# Augmentation is geometric by default -- flips and rotation, no colour or intensity jitter.
# `--aug colour` adds colour jitter and random-resized-crop on top of the geometric ops
# (`colour_crop_aug`); the colour-augmented view runs use run tag suffix `t2cs`. The default
# `--aug geometric` leaves the original loader unchanged, draw for draw.
#
# Every checkpoint and prediction file is keyworded and never overwritten.
#
# Usage:
#     python retina_dr/rdr_train.py deep --member patchbag --kw smoke --epochs 1 --limit 200
#     python retina_dr/rdr_train.py deep --member patchbag --kw anchor --epochs 8

import rdr_common as P
import rdr_common as RM
import rdr_models as RModels
import rdr_data as PRE
import rdr_data as V

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
PATCH = 224
WHOLE = 512


def load_cached(site, image_id):
    bgr = cv2.imread(PRE.cache_path(site, image_id), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f'cache miss: {site}/{image_id}')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_tensor(rgb):
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def fov_mask(rgb, bg_frac=PRE.BG_FRAC):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return g > max(1.0, float(g.max()) * bg_frac)


def geometric_aug(rgb, rng):
    """Flips and 90-degree rotations only. No colour or intensity change -- see module docstring."""
    if rng.random() < 0.5:
        rgb = rgb[:, ::-1]
    if rng.random() < 0.5:
        rgb = rgb[::-1, :]
    k = int(rng.integers(0, 4))
    if k:
        rgb = np.rot90(rgb, k)
    return np.ascontiguousarray(rgb)


AUGS = ('geometric', 'colour')
COLOUR_RANGE = (0.8, 1.2)                      # brightness, contrast, saturation factors
HUE_RANGE = 0.05                               # fraction of the hue circle
CROP_P, CROP_SCALE, CROP_RATIO = 0.5, (0.7, 1.0), (0.9, 1.1)


def colour_crop_aug(rgb, rng):
    """Colour jitter then random-resized-crop, for the `--aug colour` family only.

    Brightness, contrast and saturation factors uniform in COLOUR_RANGE and a hue shift uniform
    in +-HUE_RANGE of the circle, each with p = 1, in that order; then with p = CROP_P a crop of
    area fraction CROP_SCALE and aspect ratio CROP_RATIO (the caller resizes to the input side).
    Contrast is taken about the mean grey of the whole image, as torchvision does, so a factor
    below 1 lifts the black surround by at most 0.2 x that mean. Hue and saturation are no-ops
    on the grey (green, vessel) views. numpy/OpenCV only, uint8 in and out.
    """
    b, c, s = rng.uniform(*COLOUR_RANGE, size=3)
    h = rng.uniform(-HUE_RANGE, HUE_RANGE)
    x = rgb.astype(np.float32) * b
    m = float(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).mean()) * b
    x = np.clip((x - m) * c + m, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + h * 180.0) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s, 0, 255)
    x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if rng.random() < CROP_P:
        H, W = x.shape[:2]
        area = H * W * rng.uniform(*CROP_SCALE)
        ratio = rng.uniform(*CROP_RATIO)
        cw = min(W, int(round(np.sqrt(area * ratio))))
        ch = min(H, int(round(np.sqrt(area / ratio))))
        y0 = int(rng.integers(0, H - ch + 1))
        x0 = int(rng.integers(0, W - cw + 1))
        x = x[y0:y0 + ch, x0:x0 + cw]
    return np.ascontiguousarray(x)


def sample_patches(rgb, k, rng, patch=PATCH, min_fov=0.5, tries=20):
    """K random patches whose fundus coverage clears `min_fov`; falls back to centre crops."""
    mask = fov_mask(rgb)
    h, w = rgb.shape[:2]
    if h < patch or w < patch:
        rgb = cv2.resize(rgb, (max(w, patch), max(h, patch)), interpolation=cv2.INTER_LINEAR)
        mask = fov_mask(rgb)
        h, w = rgb.shape[:2]
    out = []
    for _ in range(k):
        for _t in range(tries):
            y = int(rng.integers(0, h - patch + 1))
            x = int(rng.integers(0, w - patch + 1))
            if mask[y:y + patch, x:x + patch].mean() >= min_fov:
                out.append(rgb[y:y + patch, x:x + patch])
                break
        else:
            cy, cx = (h - patch) // 2, (w - patch) // 2
            out.append(rgb[cy:cy + patch, cx:cx + patch])
    return out


def tile_patches(rgb, k, patch=PATCH, min_fov=0.5):
    """Deterministic eval bag: a fixed grid over the image, best-covered tiles first.

    Eval must not be stochastic, so the bag is reproducible.
    """
    mask = fov_mask(rgb)
    h, w = rgb.shape[:2]
    if h < patch or w < patch:
        rgb = cv2.resize(rgb, (max(w, patch), max(h, patch)), interpolation=cv2.INTER_LINEAR)
        mask = fov_mask(rgb)
        h, w = rgb.shape[:2]
    side = int(np.ceil(np.sqrt(k))) + 1
    ys = np.linspace(0, h - patch, side).astype(int)
    xs = np.linspace(0, w - patch, side).astype(int)
    cands = [(float(mask[y:y + patch, x:x + patch].mean()), y, x) for y in ys for x in xs]
    cands.sort(key=lambda c: -c[0])
    picked = [c for c in cands if c[0] >= min_fov][:k] or cands[:k]
    while len(picked) < k:
        picked.append(picked[-1])
    return [rgb[y:y + patch, x:x + patch] for _, y, x in picked]


# Whole-image input side. DINOv2 is patch-14, so it needs a multiple of 14 (518, not 512);
# everything else takes 512. Passed explicitly rather than inferred, so a mismatch fails at
# argument parsing instead of inside the first forward pass.
MEMBER_SIZE = {'dinov2': 518}


class RetinaSet(Dataset):
    """`transform` is an optional evaluation-time image transform.

    It receives (rgb, row, rng) and returns a modified rgb. It is applied AFTER loading and
    before any patching or resizing, so an intervention acts on the same pixels the member
    was trained to read. It is never used at train time.
    """

    def __init__(self, rows, member, train, k=8, seed=0, transform=None, size=WHOLE,
                 view='rgb', aug='geometric'):
        self.rows, self.member, self.train, self.k = rows, member, train, k
        self.view = view
        if aug not in AUGS:
            raise ValueError(f'unknown aug {aug}')
        self.aug = aug
        self.seed = seed
        self.transform = transform
        self.size = size
        if train and transform is not None:
            raise ValueError('interventions are evaluation-only')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        rgb = V.apply(self.view, load_cached(r['site'], r['image_id']), r['site'], r['image_id'])
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        if self.train:
            rgb = geometric_aug(rgb, rng)
            if self.aug == 'colour':
                rgb = colour_crop_aug(rgb, rng)
        if self.transform is not None:
            rgb = self.transform(rgb, r, rng)
        if self.member == 'patchbag':
            pats = sample_patches(rgb, self.k, rng) if self.train else tile_patches(rgb, self.k)
            x = torch.stack([to_tensor(p) for p in pats])           # [K,3,P,P]
        else:
            x = to_tensor(cv2.resize(rgb, (self.size, self.size),
                                     interpolation=cv2.INTER_AREA))
        return x, int(r['grade']), i


def read_rows(site, split=None, limit=0, limit_seed=12345):
    """Rows for a site/split, optionally capped.

    The cap takes a GRADE-STRATIFIED sample, not the head of the manifest. The manifests are
    in file order and DDR's leading rows are all grade 0, so slicing gave a single-class
    subset and an undefined AUC -- a smoke run that could not detect a broken model.
    """
    import rdr_data as D
    rows = [r for r in D.read_manifest_rows(site) if split is None or r['split'] == split]
    if limit and limit < len(rows):
        rng = np.random.default_rng(limit_seed)
        by_grade = {}
        for r in rows:
            by_grade.setdefault(int(r['grade']), []).append(r)
        per = max(1, limit // max(1, len(by_grade)))
        picked = []
        for g in sorted(by_grade):
            pool = by_grade[g]
            take = min(per, len(pool))
            picked += [pool[i] for i in rng.choice(len(pool), take, replace=False)]
        rows = picked[:limit]
    return rows


@torch.no_grad()
def predict(model, rows, member, device, k, bs, transform=None, size=None, view='rgb'):
    model.eval()
    dl = DataLoader(RetinaSet(rows, member, train=False, k=k, transform=transform,
                              size=size or MEMBER_SIZE.get(member, WHOLE), view=view),
                    batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)
    cum = np.zeros((len(rows), 4), np.float64)
    for x, _y, idx in dl:
        logits = model(x.to(device, non_blocking=True))
        cum[idx.numpy()] = torch.sigmoid(logits).float().cpu().numpy()
    return RModels.ordinal_to_proba(cum, 5)


def main_deep(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--member', required=True,
                    choices=('patchbag', 'whole') + tuple(RModels.EXTRA_VIEWS))
    ap.add_argument('--backbone', default=None, help='override the member default backbone')
    ap.add_argument('--kw', required=True, help='run keyword (part of the run tag)')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--k', type=int, default=8, help='patches per bag (patchbag only)')
    ap.add_argument('--bs', type=int, default=0, help='0 = per-member default')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--trunk-div', type=float, default=10.0,
                    help='divide the trunk lr by this (1 = flat, as the first two members used)')
    ap.add_argument('--wd', type=float, default=1e-4, help='AdamW weight decay (1e-4 = original recipe)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='cap rows per split (smoke runs)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--view', default='rgb', choices=V.VIEWS,
                    help='controlled input view (rdr_data.py views); whole-image members only')
    ap.add_argument('--extra-site', action='append', default=[], choices=('deepdrid',),
                    help='also score this OOD site')
    ap.add_argument('--blind', action='store_true',
                    help='score DDR test and the OOD sites to disk only, print nothing evaluative')
    ap.add_argument('--aug', default='geometric', choices=AUGS,
                    help='geometric = flips and rotation; colour = adds colour jitter and random-resized crop')
    args = ap.parse_args(argv)
    if args.view != 'rgb' and args.member == 'patchbag':
        ap.error('--view applies to whole-image members only')

    bs = args.bs or (16 if args.member == 'patchbag' else 24)
    run_tag = P.rdr_run_tag(f'{args.member}-{args.kw}')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train_rows = read_rows('ddr', 'train', args.limit)
    valid_rows = read_rows('ddr', 'valid', args.limit)
    print(f'{run_tag}: train={len(train_rows)} valid={len(valid_rows)} '
          f'member={args.member} bs={bs} epochs={args.epochs} device={device}', flush=True)

    model = RModels.build(args.member, backbone=args.backbone).to(device)
    # Discriminative learning rates: the pretrained trunk gets lr/trunk_div, the fresh head
    # gets lr. lr, trunk_div and wd are per-model recipe choices made on DDR valid
    # (`rdr_train.py tune`, `artifacts/tune/tune_<member>_tune.json`); --trunk-div 1 is the flat rate.
    trunk_params = [q for n, q in model.named_parameters() if n.startswith('trunk')]
    head_params = [q for n, q in model.named_parameters() if not n.startswith('trunk')]
    opt = torch.optim.AdamW(
        [{'params': trunk_params, 'lr': args.lr / args.trunk_div},
         {'params': head_params, 'lr': args.lr}], weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    lossf = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    dl = DataLoader(RetinaSet(train_rows, args.member, train=True, k=args.k, seed=args.seed,
                              size=MEMBER_SIZE.get(args.member, WHOLE), view=args.view,
                              aug=args.aug),
                    batch_size=bs, shuffle=True, num_workers=8, pin_memory=True,
                    drop_last=True, persistent_workers=True)

    P.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    history = []
    for ep in range(args.epochs):
        model.train()
        tot = n = 0
        for x, y, _ in dl:
            x = x.to(device, non_blocking=True)
            t = RModels.ordinal_targets(y, 5).to(device)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                loss = lossf(model(x), t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * len(y)
            n += len(y)
        sched.step()
        train_loss = tot / max(n, 1)
        # DDR valid: sanity and divergence abort only. Never selects a checkpoint.
        vp = predict(model, valid_rows, args.member, device, args.k, bs, view=args.view)
        vs = RM.cut_scores(np.array([int(r['grade']) for r in valid_rows]), vp, RM.RETINA_CUTS)
        history.append({'epoch': ep, 'train_loss': train_loss, 'valid_mauc': vs['mean_auc']})
        print(f'  ep{ep} loss={train_loss:.4f} valid_mAUC={vs["mean_auc"]:.4f} '
              f'({time.time() - t0:.0f}s)', flush=True)
        if not np.isfinite(train_loss):
            raise RuntimeError(f'divergence abort at epoch {ep}: loss={train_loss}')

    ckpt = P.CKPT_DIR / f'{run_tag}.pt'
    torch.save({'state_dict': model.state_dict(), 'member': args.member,
                'args': vars(args), 'history': history}, P.assert_rdr_output(ckpt))
    print(f'saved {ckpt}', flush=True)

    # ---- score every site; OOD sets are touched here for the first time ----
    preds, summary = {}, {}
    for site, split in (('ddr', 'test'), ('aptos', None), ('messidor2', None)) + tuple(
            (s, None) for s in args.extra_site):
        rows = read_rows(site, split, args.limit)
        if not rows:
            continue
        proba = predict(model, rows, args.member, device, args.k, bs, view=args.view)
        grades = np.array([int(r['grade']) for r in rows])
        preds[site] = {'ids': np.array([r['image_id'] for r in rows]),
                       'grade': grades, 'proba5': proba}
        s = RM.cut_scores(grades, proba, RM.RETINA_CUTS)
        summary[site] = s
        if args.blind:
            # nothing evaluative is printed: DDR test and the OOD sites are scored to disk only
            print(f'  {site:10s} n={len(rows):5d} (blind)', flush=True)
            continue
        print(f'  {site:10s} n={len(rows):5d} mAUC={s["mean_auc"]:.4f} '
              f'mACC={s["mean_acc"]:.4f}', flush=True)

    out = P.ARTIFACTS / 'preds' / f'pred_{run_tag}.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(P.assert_rdr_output(out),
                        **{f'{s}_{k}': v for s, d in preds.items() for k, v in d.items()})
    meta = P.ARTIFACTS / 'preds' / f'pred_{run_tag}.json'
    with open(P.assert_rdr_output(meta), 'w') as fh:
        json.dump({'run_tag': run_tag, 'args': vars(args), 'history': history,
                   'scores': summary,
                   'worst_ood_mauc': min((summary[s]['mean_auc'] for s in summary
                                          if s in ('aptos', 'messidor2')), default=None)},
                  fh, indent=2)
    print(f'wrote {out}\nwrote {meta}', flush=True)
    print(f'total {time.time() - t0:.0f}s', flush=True)


# ==========================================================================================
# rdr_train_feat.py
# ==========================================================================================
# rdr_train_feat.py -- the handcrafted-feature members (GBM / RF / SVM), CPU.
#
# Same data discipline as the deep members: fit on the DDR train split, score DDR test as the
# in-distribution arm and APTOS / Messidor-2 as held-out sites, fixed hyperparameters, no
# tuning on any site. GBM at n=300, depth 3, lr 0.05, subsample 0.8.
#
# GBM / RF / SVM share the one 28-dim feature view, so they vary the CLASSIFIER FAMILY while
# holding the input view fixed. That is the contrast the lever ordering rests on, and it is why
# all three are kept even though they are expected to correlate strongly.
#
# Outputs match the deep members' layout (pred_<run_tag>.npz with ids / grade / proba5 per
# site), so the analysis treats every member through one reader. The fitted model is kept,
# never overwritten.
#
# Usage:
#     python retina_dr/rdr_train.py feat --clf gbm --kw feat
#     python retina_dr/rdr_train.py feat --clf all --kw feat
#     python retina_dr/rdr_train.py feat --clf gbm --subset colour --kw colour-t2 --extra-site deepdrid --blind

import rdr_common as P
import rdr_common as RM


N_CLASSES = 5


def new_clf(kind):
    """Fixed classifier settings; no tuning."""
    if kind == 'gbm':
        return GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                          subsample=0.8, min_samples_leaf=10, random_state=P.SEED)
    if kind == 'rf':
        return RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=3,
                                      max_features='sqrt', class_weight='balanced',
                                      n_jobs=-1, random_state=P.SEED)
    if kind == 'svm':
        return make_pipeline(
            SimpleImputer(strategy='median'), StandardScaler(),
            SVC(C=1.0, kernel='rbf', gamma='scale', probability=True,
                class_weight='balanced', random_state=P.SEED))
    raise ValueError(kind)


WITH_LESION = False      # set by --with-lesion; appends the detector's 4 burden features
SUBSET = None            # set by --subset; keeps only these feature columns
EXTRA_SITES = ()         # set by --extra-site
BLIND = False            # set by --blind; OOD scores are written but not printed
AUG = None               # set by --aug colour; fit on features/ddr_aug-colour.npz rows

# Handcrafted feature subsets: one classifier, the
# feature view varied. Column names from rdr_features.NAMES and features/<site>_lesion.npz.
SUBSETS = {
    'colour': [f'{c}_{s}' for c in 'rgb' for s in ('mean', 'std', 'p10', 'p50', 'p90')],
    'texvessel': ['glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation',
                  'vessel_mean', 'vessel_std', 'vessel_frac'],
    'tophat': ['bright_count', 'bright_area', 'bright_mean', 'dark_count', 'dark_area',
               'dark_mean'],
    'detector': ['lesion_EX', 'lesion_HE', 'lesion_MA', 'lesion_SE'],
}


def load_site(site):
    f = P.ARTIFACTS / 'features' / f'{site}.npz'
    if not f.is_file():
        raise RuntimeError(f'features missing for {site}: run rdr_data.py features first')
    d = np.load(f, allow_pickle=True)
    X, ids, names = d['X'], d['ids'].astype(str), list(d['names'])
    if WITH_LESION:
        # The top-hat "lesion" features measured nothing (|rho| < 0.11 against
        # DDR's masks); these come from a detector trained on those masks and do track true
        # burden for EX, MA and SE. Aligned by image id, never by row order.
        lf = P.ARTIFACTS / 'features' / f'{site}_lesion.npz'
        if not lf.is_file():
            raise RuntimeError(f'lesion features missing for {site}: run rdr_train.py lesion')
        dl = np.load(lf, allow_pickle=True)
        order = {v: i for i, v in enumerate(dl['ids'].astype(str))}
        idx = np.array([order[v] for v in ids])
        X = np.concatenate([X, dl['X'][idx]], axis=1)
        names = names + [str(n) for n in dl['names']]
    X, names = _select(X, names)
    return (X, d['grade'].astype(int), ids, d['split'].astype(str), names)


def _select(X, names):
    names = [str(n) for n in names]
    if SUBSET is not None:
        cols = [names.index(n) for n in SUBSETS[SUBSET]]
        X, names = X[:, cols], [names[i] for i in cols]
    return X, names


def training_matrix():
    """DDR training rows (X, y, names). With AUG set, the K augmented copies written by
    `rdr_data.py features --aug` replace the originals; evaluation rows are never augmented."""
    if AUG is None:
        X, y, _ids, split, names = load_site('ddr')
        tr = split == 'train'
        return X[tr], y[tr], names
    if WITH_LESION:
        raise RuntimeError('--aug has no lesion-feature counterpart')
    f = P.ARTIFACTS / 'features' / f'ddr_aug-{AUG}.npz'
    if not f.is_file():
        raise RuntimeError(f'augmented features missing: run rdr_data.py features --aug {AUG}')
    d = np.load(f, allow_pickle=True)
    X, names = _select(d['X'], d['names'])
    return X, d['grade'].astype(int), names


def proba5(clf, X):
    """predict_proba padded to all 5 ICDR columns, in case a class is absent from training."""
    p = clf.predict_proba(X)
    out = np.zeros((len(X), N_CLASSES), np.float64)
    for j, cls in enumerate(clf.classes_):
        out[:, int(cls)] = p[:, j]
    s = out.sum(axis=1, keepdims=True)
    return out / np.clip(s, 1e-12, None)


def run_one(kind, kw):
    run_tag = P.rdr_run_tag(f'{kind}-{kw}')
    X, y, ids, split, names = load_site('ddr')
    Xtr, ytr, names_tr = training_matrix()
    assert names_tr == names
    print(f'{run_tag}: train={len(ytr)} features={X.shape[1]} aug={AUG}', flush=True)

    clf = new_clf(kind)
    fit_kw = {}
    if kind == 'gbm':       # RF and SVM already carry class_weight='balanced'
        fit_kw['sample_weight'] = compute_sample_weight('balanced', ytr)
    clf.fit(Xtr, ytr, **fit_kw)

    P.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({'clf': clf, 'names': names, 'kind': kind},
                P.assert_rdr_output(P.CKPT_DIR / f'{run_tag}.joblib'))

    preds, summary = {}, {}
    te = split == 'test'
    for site, Xs, ys, idss in (('ddr', X[te], y[te], ids[te]),) + tuple(
            (s,) + load_site(s)[:3] for s in ('aptos', 'messidor2') + EXTRA_SITES):
        p = proba5(clf, Xs)
        preds[site] = {'ids': idss, 'grade': ys, 'proba5': p}
        sc = RM.cut_scores(ys, p, RM.RETINA_CUTS)
        summary[site] = sc
        if BLIND and site != 'ddr':
            print(f'  {site:10s} n={len(ys):5d} (blind)', flush=True)
            continue
        print(f'  {site:10s} n={len(ys):5d} mAUC={sc["mean_auc"]:.4f} '
              f'mACC={sc["mean_acc"]:.4f}', flush=True)

    out = P.ARTIFACTS / 'preds' / f'pred_{run_tag}.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(P.assert_rdr_output(out),
                        **{f'{s}_{k}': v for s, d in preds.items() for k, v in d.items()})
    with open(P.assert_rdr_output(P.ARTIFACTS / 'preds' / f'pred_{run_tag}.json'), 'w') as fh:
        json.dump({'run_tag': run_tag, 'kind': kind, 'scores': summary,
                   'subset': SUBSET, 'names': names, 'aug': AUG,
                   'worst_ood_mauc': min(summary[s]['mean_auc'] for s in summary
                                         if s in ('aptos', 'messidor2'))}, fh, indent=2)
    print(f'wrote {out}', flush=True)

    if hasattr(clf, 'feature_importances_'):
        order = np.argsort(clf.feature_importances_)[::-1][:8]
        print('  top features: ' + ', '.join(
            f'{names[i]}={clf.feature_importances_[i]:.3f}' for i in order), flush=True)


def main_feat(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--clf', required=True, choices=('gbm', 'rf', 'svm', 'all'))
    ap.add_argument('--with-lesion', action='store_true',
                    help='append the 4 detector lesion-burden features')
    ap.add_argument('--subset', choices=tuple(SUBSETS), default=None,
                    help='keep only this feature subset (detector implies --with-lesion)')
    ap.add_argument('--extra-site', action='append', default=[], choices=('deepdrid',))
    ap.add_argument('--blind', action='store_true', help='do not print OOD scores')
    ap.add_argument('--kw', required=True)
    ap.add_argument('--aug', choices=('colour',), default=None,
                    help='fit on the augmented training features of rdr_data.py features --aug')
    args = ap.parse_args(argv)
    global WITH_LESION, SUBSET, EXTRA_SITES, BLIND, AUG
    WITH_LESION = args.with_lesion or args.subset == 'detector'
    SUBSET, EXTRA_SITES, BLIND, AUG = args.subset, tuple(args.extra_site), args.blind, args.aug
    kinds = ('gbm', 'rf', 'svm') if args.clf == 'all' else (args.clf,)
    for k in kinds:
        run_one(k, args.kw)


# ==========================================================================================
# rdr_lesion_seg.py
# ==========================================================================================
# rdr_lesion_seg.py -- a real lesion detector, so the handcrafted view can carry real lesion features.
#
# Why this exists. The handcrafted view's "lesion" features are top-hat blob counts, and
# measuring them against DDR's masks showed they carry no lesion signal at all: every Spearman
# correlation with true per-class lesion area is below 0.11 in magnitude, and the microaneurysm
# one is significantly negative. Yet both tree families place four of their eight most-used
# features there. So the view is a fine-scale local-contrast detector that merely resembles a
# lesion reader.
#
# DDR ships pixel masks for four lesion classes (EX hard exudates, HE haemorrhages, MA
# microaneurysms, SE soft exudates) over 757 images. This trains a small U-Net on them and
# emits, for every image in every site, the predicted area fraction per class. Those four
# numbers are lesion features by construction, and the same proxy-versus-truth correlation that
# condemned the top-hat features is reported for them as a gate: a detector that does not beat
# the proxies has not earned its place.
#
# Train on the DDR lesion train split, monitor on valid, report on the held-out test split.
# The grading labels are never used here, and the OOD sites are only ever inferred on -- so a
# lesion feature cannot smuggle in site or grade information.
#
# Usage:
#     python retina_dr/rdr_train.py lesion --kw les --epochs 40
#     python retina_dr/rdr_train.py lesion --kw les --infer         # emit features for every site

import rdr_common as P
import rdr_data as PRE

SIZE = 512
CLASSES = PRE.LESION_CLASSES          # EX, HE, MA, SE


def lesion_rows(split):
    """(stem, image_path, label_dir) for one DDR lesion split."""
    img_dir, lab_dir = PRE.ddr_lesion_dirs()[split]
    if not img_dir.is_dir():
        return []
    return [(os.path.splitext(n)[0], str(img_dir / n), str(lab_dir))
            for n in sorted(os.listdir(img_dir))]


class LesionSet(Dataset):
    """Images and masks through the SAME crop and resize, so they stay in register."""

    def __init__(self, rows, train):
        self.rows, self.train = rows, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        stem, ipath, lab_dir = self.rows[i]
        bgr = cv2.imread(ipath, cv2.IMREAD_COLOR)
        y0, y1, x0, x1 = PRE.fundus_bbox(bgr)
        rgb = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        ms = []
        for cls in CLASSES:
            src = None
            for ext in ('.tif', '.png', '.jpg'):
                cand = os.path.join(lab_dir, cls, stem + ext)
                if os.path.exists(cand):
                    src = cand
                    break
            m = cv2.imread(src, cv2.IMREAD_GRAYSCALE) if src else None
            m = np.zeros(bgr.shape[:2], np.uint8) if m is None else m
            m = cv2.resize((m[y0:y1, x0:x1] > 0).astype(np.uint8), (SIZE, SIZE),
                           interpolation=cv2.INTER_NEAREST)
            ms.append(m)
        mask = np.stack(ms).astype(np.float32)
        if self.train:
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                rgb, mask = rgb[:, ::-1], mask[:, :, ::-1]
            if rng.random() < 0.5:
                rgb, mask = rgb[::-1], mask[:, ::-1]
            rgb, mask = np.ascontiguousarray(rgb), np.ascontiguousarray(mask)
        x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(mask.copy())


def build_unet():
    from monai.networks.nets import UNet
    return UNet(spatial_dims=2, in_channels=3, out_channels=len(CLASSES),
                channels=(32, 64, 128, 256, 320), strides=(2, 2, 2, 2),
                num_res_units=2, dropout=0.1)


def dice_per_class(logits, target, eps=1e-6):
    p = (torch.sigmoid(logits) > 0.5).float()
    dims = (0, 2, 3)
    inter = (p * target).sum(dims)
    denom = p.sum(dims) + target.sum(dims)
    return ((2 * inter + eps) / (denom + eps)).detach().cpu().numpy()


@torch.no_grad()
def area_fractions(model, rows_site, device, bs=8, soft=True):
    """Predicted lesion burden per class, for a list of (site, image_id).

    `soft` returns the MEAN SIGMOID rather than the area above 0.5. Lesions occupy on the
    order of 1e-4 of the pixels, so a detector can rank images correctly while never crossing
    0.5 -- in which case a thresholded area is identically zero and useless as a feature. The
    soft mean keeps the ranking, needs no threshold, and degrades gracefully when the detector
    is weak. Dice is still reported against the thresholded mask, since that is what Dice
    means.
    """
    model.eval()
    out = np.zeros((len(rows_site), len(CLASSES)), np.float32)
    batch, idx = [], []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        pr = torch.sigmoid(model(x))
        pr = pr.mean(dim=(2, 3)) if soft else (pr > 0.5).float().mean(dim=(2, 3))
        pr = pr.cpu().numpy()
        for j, k in enumerate(idx):
            out[k] = pr[j]
        batch.clear()
        idx.clear()

    for i, (site, image_id) in enumerate(rows_site):
        bgr = cv2.imread(PRE.cache_path(site, image_id), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        batch.append(torch.from_numpy(x.transpose(2, 0, 1).copy()))
        idx.append(i)
        if len(batch) == bs:
            flush()
    flush()
    return out


def main_lesion(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--kw', required=True)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--pos-weight', type=float, nargs=4, default=[641.0, 708.0, 500.0, 500.0],
                    help='per-class BCE pos_weight for EX HE MA SE (measured ratios, capped)')
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--infer', action='store_true', help='skip training, load ckpt, emit features')
    args = ap.parse_args(argv)

    run_tag = P.rdr_run_tag(f'lesion-{args.kw}')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ck_path = P.CKPT_DIR / f'{run_tag}.pt'
    model = build_unet().to(device)

    if not args.infer:
        tr, va, te = lesion_rows('train'), lesion_rows('valid'), lesion_rows('test')
        print(f'{run_tag}: train={len(tr)} valid={len(va)} test={len(te)}', flush=True)
        dl = DataLoader(LesionSet(tr, True), batch_size=args.bs, shuffle=True,
                        num_workers=6, pin_memory=True, drop_last=True)
        vl = DataLoader(LesionSet(va, False), batch_size=args.bs, num_workers=6)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        # Lesion pixels are a vanishing fraction of the image, MEASURED on the training split
        # as EX 0.156%, HE 0.141%, MA 0.0135%, SE 0.0042% -- neg/pos ratios of 641, 708, 7407
        # and 23766. A flat pos_weight of 20 (the first attempt) was two to three orders of
        # magnitude short and the model collapsed to all-background by epoch 10. Weights are
        # now the measured per-class ratios, capped so the rarest class cannot destabilise the
        # update.
        pw = torch.tensor(args.pos_weight, device=device).view(len(CLASSES), 1, 1)
        bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
        for ep in range(args.epochs):
            model.train()
            tot = n = 0
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    lg = model(x)
                    p = torch.sigmoid(lg)
                    # Per-image, per-class soft Dice with a small epsilon. Summing over the
                    # batch and smoothing by 1 (the first attempt) made the term nearly flat
                    # at p=0, so it supplied almost no gradient to escape all-background.
                    inter = (p * y).sum((2, 3))
                    denom = p.sum((2, 3)) + y.sum((2, 3))
                    dice = (1 - (2 * inter + 1e-6) / (denom + 1e-6)).mean()
                    loss = bce(lg, y) + dice
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += float(loss) * len(x)
                n += len(x)
            sched.step()
            if ep % 5 == 0 or ep == args.epochs - 1:
                model.eval()
                ds = []
                with torch.no_grad():
                    for x, y in vl:
                        ds.append(dice_per_class(model(x.to(device)), y.to(device)))
                dm = np.mean(ds, axis=0)
                print(f'  ep{ep} loss={tot / max(n,1):.4f} valid dice ' +
                      ' '.join(f'{c}={v:.3f}' for c, v in zip(CLASSES, dm)), flush=True)
        P.CKPT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': model.state_dict(), 'classes': list(CLASSES)},
                   P.assert_rdr_output(ck_path))
        print(f'saved {ck_path}', flush=True)

        # held-out test Dice
        tl = DataLoader(LesionSet(te, False), batch_size=args.bs, num_workers=6)
        model.eval()
        ds = []
        with torch.no_grad():
            for x, y in tl:
                ds.append(dice_per_class(model(x.to(device)), y.to(device)))
        dm = np.mean(ds, axis=0)
        print('held-out test dice ' + ' '.join(f'{c}={v:.3f}' for c, v in zip(CLASSES, dm)),
              flush=True)
        with open(P.assert_rdr_output(P.ARTIFACTS / f'lesion_seg_{run_tag}.json'), 'w') as fh:
            json.dump({'run_tag': run_tag,
                       'test_dice': dict(zip(CLASSES, [float(v) for v in dm]))}, fh, indent=2)
    else:
        model.load_state_dict(torch.load(ck_path, map_location='cpu',
                                         weights_only=False)['state_dict'])

    # ---- emit lesion features for every site, and gate them against the top-hat proxies ----
    import rdr_data as D
    feats = {}
    for site in ('ddr', 'aptos', 'messidor2'):
        rows = D.read_manifest_rows(site)
        if not rows:
            continue
        af = area_fractions(model, [(r['site'], r['image_id']) for r in rows], device)
        feats[site] = af
        out = P.ARTIFACTS / 'features' / f'{site}_lesion.npz'
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(P.assert_rdr_output(out), X=af,
                            ids=np.array([r['image_id'] for r in rows]),
                            names=np.array([f'lesion_{c}' for c in CLASSES]))
        print(f'{site:10s} lesion features {af.shape} -> {out}', flush=True)

    # Gate: do the DETECTED areas track true lesion area better than the top-hat proxies did?
    from scipy.stats import spearmanr
    te = lesion_rows('test')
    if te:
        true_area, pred_area = [], []
        for stem, ipath, lab_dir in te:
            bgr = cv2.imread(ipath, cv2.IMREAD_COLOR)
            y0, y1, x0, x1 = PRE.fundus_bbox(bgr)
            tr_row = []
            for cls in CLASSES:
                src = None
                for ext in ('.tif', '.png', '.jpg'):
                    cand = os.path.join(lab_dir, cls, stem + ext)
                    if os.path.exists(cand):
                        src = cand
                        break
                m = cv2.imread(src, cv2.IMREAD_GRAYSCALE) if src else None
                if m is None:
                    tr_row.append(0.0)
                    continue
                m = cv2.resize((m[y0:y1, x0:x1] > 0).astype(np.uint8), (SIZE, SIZE),
                               interpolation=cv2.INTER_NEAREST)
                tr_row.append(float(m.mean()))
            true_area.append(tr_row)
        pa = area_fractions(model, [('ddr', s + '.jpg') for s, _, _ in te], device)
        true_area = np.array(true_area)
        print('\ndetector vs TRUE lesion area, held-out DDR test (the proxies scored |rho|<0.11):',
              flush=True)
        for j, cls in enumerate(CLASSES):
            r, pv = spearmanr(pa[:, j], true_area[:, j])
            print(f'  lesion_{cls:2s}  rho={r:+.3f}  p={pv:.2e}', flush=True)


# ==========================================================================================
# tune
# ==========================================================================================
# Optuna recipe search for one deep member. Objective = DDR valid mAUC at the last epoch; DDR
# test and every OOD site stay unread (`--blind`). Each trial is one `rdr_train.py deep`
# subprocess with run tag retina-<member>-<kw>-tNN; checkpoints and predictions are keyworded
# and never overwritten. Trial 0 is the original recipe (enqueued).
#
# Usage:
#     python retina_dr/rdr_train.py tune --member patchbag --kw tune --trials 10 --device cuda:0

import subprocess
from pathlib import Path

TUNE_HERE = Path(__file__).resolve().parent
TUNE_DIR = P.ARTIFACTS / 'tune'
TUNE_SEED = 20261013
ORIGINAL = {'lr': 3e-4, 'wd': 1e-4}             # the recipe of the untuned members
ORIGINAL_DIV = {'patchbag': 1.0, 'whole': 1.0}      # flat rate for the first two members


def run_trial(member, tag_kw, lr, div, wd, epochs, device, limit):
    cmd = [sys.executable, str(TUNE_HERE / 'rdr_train.py'), 'deep', '--member', member, '--kw', tag_kw,
           '--epochs', str(epochs), '--lr', f'{lr:.6g}', '--trunk-div', f'{div:g}',
           '--wd', f'{wd:.6g}', '--device', device, '--blind']
    if limit:
        cmd += ['--limit', str(limit)]
    print('+', ' '.join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(TUNE_HERE))
    meta = P.ARTIFACTS / 'preds' / f'pred_retina-{member}-{tag_kw}.json'
    if rc != 0 or not meta.is_file():
        print(f'  trial failed rc={rc} ({time.time() - t0:.0f}s)', flush=True)
        return None, time.time() - t0
    hist = json.load(open(meta))['history']
    return float(hist[-1]['valid_mauc']), time.time() - t0


def main_tune(argv=None):
    import optuna

    ap = argparse.ArgumentParser()
    ap.add_argument('--member', required=True, choices=('patchbag', 'whole', 'convnext', 'dinov2', 'effnet'))
    ap.add_argument('--kw', required=True)
    ap.add_argument('--trials', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--limit', type=int, default=0, help='smoke runs only')
    args = ap.parse_args(argv)

    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    db = TUNE_DIR / f'{args.member}_{args.kw}.db'
    study = optuna.create_study(study_name=f'{args.member}-{args.kw}', direction='maximize',
                                storage=f'sqlite:///{db}', load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=TUNE_SEED))
    if len(study.trials) == 0:
        study.enqueue_trial(dict(ORIGINAL, trunk_div=ORIGINAL_DIV.get(args.member, 10.0)))

    def objective(trial):
        lr = trial.suggest_float('lr', 5e-5, 1e-3, log=True)
        div = trial.suggest_categorical('trunk_div', [1.0, 3.0, 10.0, 30.0])
        wd = trial.suggest_float('wd', 1e-6, 1e-3, log=True)
        tag_kw = f'{args.kw}-t{trial.number:02d}'
        val, secs = run_trial(args.member, tag_kw, lr, div, wd, args.epochs, args.device, args.limit)
        trial.set_user_attr('run_tag', f'retina-{args.member}-{tag_kw}')
        trial.set_user_attr('seconds', secs)
        if val is None:
            raise optuna.TrialPruned()
        print(f'  trial {trial.number} lr={lr:.2e} div={div:g} wd={wd:.1e} '
              f'valid_mAUC={val:.4f} ({secs:.0f}s)', flush=True)
        return val

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)
    done = [t for t in study.trials if t.value is not None]
    best = study.best_trial
    out = {'member': args.member, 'kw': args.kw, 'objective': 'ddr valid mAUC, last epoch',
           'epochs': args.epochs, 'n_trials': len(study.trials), 'n_completed': len(done),
           'best': {'number': best.number, 'value': best.value, 'params': best.params,
                    'run_tag': best.user_attrs.get('run_tag')},
           'original_trial0': {
               'value': study.trials[0].value, 'params': study.trials[0].params,
                                 'run_tag': study.trials[0].user_attrs.get('run_tag')},
           'trials': [{'number': t.number, 'value': t.value, 'params': t.params,
                       'run_tag': t.user_attrs.get('run_tag'), 'seconds': t.user_attrs.get('seconds'),
                       'state': str(t.state)} for t in study.trials]}
    summ = TUNE_DIR / f'tune_{args.member}_{args.kw}.json'
    summ.write_text(json.dumps(out, indent=1))
    print(f'best trial {best.number} valid_mAUC={best.value:.4f} {best.params} -> {summ}', flush=True)


COMMANDS = {'deep': main_deep, 'feat': main_feat, 'lesion': main_lesion, 'tune': main_tune}


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: {os.path.basename(sys.argv[0])} {'|'.join(COMMANDS)} [options]")
    COMMANDS[sys.argv[1]](sys.argv[2:])
