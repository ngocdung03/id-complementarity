"""Paths, run tags, scoring and combination functions, strict alignment, pair statistics and
net correction shared by every script; the `operator` subcommand writes the operator table at
w = 0.15, 0.20 and 0.25.

Usage:
    python retina_dr/rdr_common.py operator --anchor retina-patchbag-anchor --members <tags> --kw all7
"""
from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
import argparse
import hashlib
import itertools
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score


# ==========================================================================================
# paths and run tags
# ==========================================================================================
#
# Every path derives from one root, `RDR_ROOT` (environment variable; default: the repository
# root). Under it:
#
#     data/       the downloaded datasets (never committed)
#     artifacts/  manifests, caches, features, checkpoints, predictions and analysis outputs
#
# The output guard is `assert_rdr_output`: assert output IS inside the artifact namespace, so a
# mistyped --out cannot land elsewhere.
#

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

ROOT = Path(os.environ.get('RDR_ROOT', os.path.dirname(HERE))).resolve()
CARE_ROOT = ROOT
ARTIFACTS = ROOT / 'artifacts'
CACHE_DIR = ARTIFACTS / 'cache'
CKPT_DIR = ARTIFACTS / 'ckpt'
LOG_DIR = ARTIFACTS / 'logs'
RUN_PREFIX = 'retina'
SEED = 42

# Raw downloads. Never committed -- licences restrict redistribution.
EXTERNAL = ROOT / 'data'

# Site roots, as the 8 October 2026 downloads unpacked.
DDR_ROOT = EXTERNAL / 'ddr' / 'extracted' / 'DDR-dataset' / 'DDR-dataset'
DDR_GRADING = DDR_ROOT / 'DR_grading'
DDR_LESION = DDR_ROOT / 'lesion_segmentation'
APTOS_ROOT = EXTERNAL / 'aptos2019' / 'extracted'
MESSIDOR_ROOT = EXTERNAL / 'messidor2' / 'extracted'

# ICDR grade endpoints: referable DR (>= 2) and vision-threatening DR (>= 3), read off a
# 5-class vector.
CUTS = (2, 3)
N_CLASSES = 5

SITES = ('ddr', 'aptos', 'messidor2', 'idrid')
OOD_SITES = ('aptos', 'messidor2', 'idrid')


def assert_rdr_output(path) -> Path:
    """Fail fast if a retina output would land outside the artifact namespace."""
    resolved = Path(path).resolve()
    root = ARTIFACTS.resolve()
    if root == resolved or root in resolved.parents:
        return resolved
    raise RuntimeError(f"retina output must be under {root}: {resolved}")


def resolve_keyword(kw_arg):
    """Resolve the mandatory per-run keyword that gets appended to a run_tag.

    Pass --kw on the command line. If omitted AND stdin is a TTY, prompt; if
    omitted in a detached/non-TTY run, exit with an error (never block). The
    keyword is sanitized to lowercase kebab-case before use.
    """
    kw = (kw_arg or '').strip()
    if not kw and sys.stdin.isatty():
        kw = input('Run keyword (brief, required): ').strip()
    kw = re.sub(r'[^A-Za-z0-9]+', '-', kw).strip('-').lower()
    if not kw:
        sys.exit('ERROR: a run keyword is required (pass --kw KEYWORD).')
    return kw


def setup_logger(run_tag):
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger(run_tag)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s  %(message)s')
    fh = logging.FileHandler(f'{LOG_DIR}/{run_tag}.log')
    sh = logging.StreamHandler()
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def rdr_run_tag(kw_arg) -> str:
    """Sanitize --kw and prefix it as a retina run."""
    return f"{RUN_PREFIX}-{resolve_keyword(kw_arg)}"


def rdr_logger(run_tag):
    """Run logger writing into the retina artifact namespace."""
    return setup_logger(run_tag)


# ==========================================================================================
# scoring and combination
# ==========================================================================================
# Cut-point scoring, generic over class count.
#
# Index convention: a cut `c` on a K-class vector means "label index >= c", scored as
# sum(proba[:, c:]). Retina grades 0..4 occupy indices 0..4, so referable DR and
# vision-threatening DR are cuts (2, 3).
#
# Decisions are taken at a fixed 0.5; no threshold is tuned
# on any site.

RETINA_CUTS = (2, 3)    # referable DR, vision-threatening DR


def _safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float('nan')


def cut_scores(labels, proba, cuts):
    """Score a K-class probability matrix at `cuts`, in index space.

    labels: integer label INDEX per case (0-based).
    proba:  [N, K], rows summing to 1.
    Returns the per-task and mean AUC / ACC, keyed t1..tn in the order of `cuts`.
    """
    labels = np.asarray(labels)
    proba = np.asarray(proba, np.float64)
    out, aucs, accs = {}, [], []
    for i, c in enumerate(cuts, start=1):
        s = proba[:, c:].sum(axis=1)
        y = (labels >= c).astype(int)
        auc = _safe_auc(y, s)
        acc = float(accuracy_score(y, (s >= 0.5).astype(int)))
        out[f'auc_t{i}'], out[f'acc_t{i}'] = auc, acc
        aucs.append(auc)
        accs.append(acc)
    kept = [a for a in aucs if not np.isnan(a)]
    out['mean_auc'] = float(np.mean(kept)) if kept else float('nan')
    out['mean_acc'] = float(np.mean(accs))
    return out


# Convention: w goes to the FIRST argument. Call as geo_blend(p_member, p_anchor, 0.15) so the
# ADDED member receives w.
EPS = 1e-9   # additive floor before the logarithm


def geo_blend(p_a, p_b, w):
    """Weighted geometric mean, renormalised: w goes to `p_a` (the added member)."""
    p_a = np.asarray(p_a, np.float64)
    p_b = np.asarray(p_b, np.float64)
    out = np.power(p_a + EPS, w) * np.power(p_b + EPS, 1.0 - w)
    return out / out.sum(axis=1, keepdims=True)


def arith_blend(p_a, p_b, w):
    """Weighted arithmetic mean, w to `p_a` (the added member)."""
    p_a = np.asarray(p_a, np.float64)
    p_b = np.asarray(p_b, np.float64)
    out = w * p_a + (1.0 - w) * p_b
    return out / out.sum(axis=1, keepdims=True)




# ==========================================================================================
# alignment, operator table, pair statistics and net correction
# ==========================================================================================
# Reads only the stored per-member prediction files. Pair machinery: for one cut, every
# (positive, negative) pair gets a correctness of 1, 0.5 or 0 (positive above, tie, below); its
# mean over all pairs is the AUC. Net correction is the share of anchor-misordered pairs the
# candidate orders correctly minus the share of anchor-correct pairs it misorders.

P = RM = sys.modules[__name__]

ANCHOR = 'retina-patchbag-anchor'
# The seven candidates of Table 1, in table order.
MEMBERS = ('retina-convnext-tune-t08', 'retina-whole-view', 'retina-effnet-tune-t03',
           'retina-dinov2-tune-t02', 'retina-svm-feat', 'retina-rf-feat', 'retina-gbm-feat')
ID_SITE = 'ddr'
W = 0.15


def align_strict(anchor_site, member_site):
    """Intersect on image id; require unique ids and identical grades on the overlap.

    Returns (ids, grade, p_anchor, p_member, counts). Never relies on row order.
    """
    ia, ib = anchor_site['ids'], member_site['ids']
    if len(set(ia)) != len(ia) or len(set(ib)) != len(ib):
        raise ValueError('duplicate image ids')
    common = np.intersect1d(ia, ib)
    if common.size == 0:
        raise ValueError('no shared image ids')
    fa = {v: i for i, v in enumerate(ia)}
    fb = {v: i for i, v in enumerate(ib)}
    ai = np.array([fa[c] for c in common])
    bi = np.array([fb[c] for c in common])
    ga, gb = anchor_site['grade'][ai], member_site['grade'][bi]
    if not np.array_equal(ga, gb):
        raise ValueError(f'grade mismatch on {int((ga != gb).sum())} shared ids')
    counts = {'n_anchor': int(len(ia)), 'n_member': int(len(ib)), 'n_aligned': int(common.size)}
    return common, ga, anchor_site['proba5'][ai], member_site['proba5'][bi], counts


W_PRIMARY = 0.15
W_EXTRA = (0.20, 0.25)
OOD = ('aptos', 'messidor2')
PRED_DIR = P.ARTIFACTS / 'preds'


def load_pred(run_tag):
    """-> {site: {'ids', 'grade', 'proba5'}} for every site the run scored."""
    f = PRED_DIR / f'pred_{run_tag}.npz'
    if not f.is_file():
        raise RuntimeError(f'missing predictions: {f}')
    d = np.load(f, allow_pickle=True)
    sites = sorted({k.rsplit('_', 1)[0] for k in d.files})
    out = {}
    for s in sites:
        if f'{s}_proba5' not in d.files:
            continue
        out[s] = {'ids': d[f'{s}_ids'].astype(str), 'grade': d[f'{s}_grade'].astype(int),
                  'proba5': d[f'{s}_proba5'].astype(np.float64)}
    return out


def align(a, b):
    """Intersect two members on image id, per site. Never assume row order matches."""
    out = {}
    for site in sorted(set(a) & set(b)):
        ia, ib = a[site]['ids'], b[site]['ids']
        common = np.intersect1d(ia, ib)
        fa = {v: i for i, v in enumerate(ia)}
        fb = {v: i for i, v in enumerate(ib)}
        ai = [fa[c] for c in common]
        bi = [fb[c] for c in common]
        out[site] = (a[site]['grade'][ai], a[site]['proba5'][ai], b[site]['proba5'][bi])
    return out


def operator_table(anchor_tag, member_tags):
    anchor = load_pred(anchor_tag)
    rows = {'anchor': {s: RM.cut_scores(anchor[s]['grade'], anchor[s]['proba5'],
                                        RM.RETINA_CUTS) for s in anchor}}
    base_id = rows['anchor'].get('ddr', {}).get('mean_auc', float('nan'))
    base_ood = [rows['anchor'][s]['mean_auc'] for s in OOD if s in rows['anchor']]
    for tag in member_tags:
        mem = load_pred(tag)
        al = align(anchor, mem)
        entry = {}
        for w in (W_PRIMARY,) + W_EXTRA:
            for op, fn in (('geo', RM.geo_blend), ('arith', RM.arith_blend)):
                per = {}
                for site, (g, pa, pb) in al.items():
                    per[site] = RM.cut_scores(g, fn(pb, pa, w), RM.RETINA_CUTS)
                ood = [per[s]['mean_auc'] for s in OOD if s in per]
                entry[f'{op}_{w:.2f}'] = {
                    'per_site': per,
                    'd_id': per['ddr']['mean_auc'] - base_id if 'ddr' in per else None,
                    'd_worst': (min(ood) - min(base_ood)) if ood and base_ood else None,
                    'd_ood_mean': (float(np.mean(ood)) - float(np.mean(base_ood)))
                    if ood and base_ood else None}
        standalone = {s: RM.cut_scores(mem[s]['grade'], mem[s]['proba5'], RM.RETINA_CUTS)
                      for s in mem}
        entry['standalone'] = standalone
        rows[tag] = entry
    return rows, base_id, base_ood


DEEP = ('retina-convnext-tune-t08', 'retina-whole-view', 'retina-effnet-tune-t03',
        'retina-dinov2-tune-t02')
HANDCRAFTED = ('retina-svm-feat', 'retina-rf-feat', 'retina-gbm-feat')
assert set(DEEP + HANDCRAFTED) == set(MEMBERS)
OUT_DIR = P.ARTIFACTS / 'mechanism'
MECH_SEED = 20261013       # bootstrap and permutation seed of the analyses


def load_site(tag, site):
    """One site of one prediction file; reads only that site's arrays."""
    f = PRED_DIR / f'pred_{tag}.npz'
    if not f.is_file():
        raise RuntimeError(f'missing predictions: {f}')
    d = np.load(f, allow_pickle=False)
    return {'ids': d[f'{site}_ids'].astype(str), 'grade': d[f'{site}_grade'].astype(int),
            'proba5': d[f'{site}_proba5'].astype(np.float64)}


def aligned(member, site, anchor=ANCHOR):
    """(ids, grade, p_anchor, p_member) on the shared ids (strict alignment)."""
    ids, g, pa, pm, _ = align_strict(load_site(anchor, site), load_site(member, site))
    return ids, g, pa, pm


def cut_score(proba, c):
    return np.asarray(proba, np.float64)[:, c:].sum(axis=1)


def pair_correct(s_pos, s_neg):
    """[n_pos, n_neg] correctness matrix: 1 if positive above, 0.5 on a tie, 0 below."""
    diff = s_pos[:, None] - s_neg[None, :]
    return (diff > 0) + 0.5 * (diff == 0)


def split_by_cut(labels, c):
    y = np.asarray(labels) >= c
    return np.flatnonzero(y), np.flatnonzero(~y)


def spearman_exact(x, y, alternative='two-sided'):
    """Spearman rho with an exact permutation p over all orderings of y (n <= 8)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) > 8:
        raise ValueError('exact permutation is only enumerated for n <= 8')
    rho = float(spearmanr(x, y)[0])
    perm = np.array([spearmanr(x, y[list(p)])[0]
                     for p in itertools.permutations(range(len(y)))])
    if alternative == 'greater':
        p = float(np.mean(perm >= rho - 1e-12))
    else:
        p = float(np.mean(np.abs(perm) >= abs(rho) - 1e-12))
    return {'rho': rho, 'p_exact': p, 'n': int(len(x)), 'n_perm': int(len(perm)),
            'alternative': alternative}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def input_hashes(tags):
    return {t: sha256(PRED_DIR / f'pred_{t}.npz') for t in tags}


def out_path(stem, run_tag, suffix='json'):
    """Versioned output under mechanism/, refusing to overwrite."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = P.assert_rdr_output(OUT_DIR / f'{stem}_{run_tag}.{suffix}')
    if out.exists():
        raise SystemExit(f'refusing to overwrite {out}; choose a new --kw')
    return out


def write_json(out, payload):
    with open(out, 'w') as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f'wrote {out}')


def rescue_cut(g, pa, pm, c):
    pos, neg = split_by_cut(g, c)
    sa, sm = cut_score(pa, c), cut_score(pm, c)
    ca = pair_correct(sa[pos], sa[neg])
    cm = pair_correct(sm[pos], sm[neg])
    a_wrong = 1.0 - ca
    rescue = float((a_wrong * cm).sum() / a_wrong.sum())
    harm = float((ca * (1.0 - cm)).sum() / ca.sum())
    return {'cut': int(c), 'n_pairs': int(ca.size), 'anchor_wrong_pairs': float(a_wrong.sum()),
            'rescue_rate': rescue, 'harm_rate': harm, 'net': rescue - harm}


def main_operator(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--anchor', required=True)
    ap.add_argument('--members', required=True, help='comma-separated run tags')
    ap.add_argument('--kw', default='analysis')
    args = ap.parse_args(argv)
    member_tags = [m.strip() for m in args.members.split(',') if m.strip()]

    table, base_id, base_ood = operator_table(args.anchor, member_tags)
    print(f'anchor {args.anchor}: ID mAUC={base_id:.4f} '
          f'worst-OOD={min(base_ood):.4f}' if base_ood else '')
    for tag in member_tags:
        st = table[tag]['standalone']
        blk = table[tag][f'geo_{W_PRIMARY:.2f}']
        ar = table[tag][f'arith_{W_PRIMARY:.2f}']
        print(f'{tag:28s} standalone ID={st.get("ddr", {}).get("mean_auc", float("nan")):.4f} '
              f'| geo dID={blk["d_id"]:+.4f} dworst={blk["d_worst"]:+.4f} '
              f'| arith dID={ar["d_id"]:+.4f} dworst={ar["d_worst"]:+.4f}')


    out = P.ARTIFACTS / f'analysis_{P.rdr_run_tag(args.kw)}.json'
    with open(P.assert_rdr_output(out), 'w') as fh:
        json.dump({'anchor': args.anchor, 'members': member_tags, 'operator_table': table}, fh, indent=2, default=str)
    print(f'wrote {out}')


COMMANDS = {'operator': main_operator}


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: {os.path.basename(sys.argv[0])} {'|'.join(COMMANDS)} [options]")
    COMMANDS[sys.argv[1]](sys.argv[2:])
