#!/usr/bin/env python
"""Predictions on the two later OOD sites, the view runs' IDRiD predictions, the DDR bootstrap
draws for the view runs, and the view-run statistics.

  site prepare   manifest, preprocessing cache, handcrafted and lesion features of DeepDRiD or
                 IDRiD (`--site`; GPU for the lesion features)
  site predict   every candidate's grade probabilities on that site (GPU for deep candidates)
  idrid-views    the twelve view runs' grade probabilities on IDRiD (GPU)
  boot           grade-stratified image-level bootstrap of net correction and mAUC on DDR test
                 for the view runs (draws read by rdr_report.py)
  viewstats      per view run: net correction and mAUC on DDR test, geometric contribution on
                 APTOS, Messidor-2 and DeepDRiD; one-way F over the four views with a
                 permutation p (`--suffix t2cs` for the colour-augmented runs)

Every output is a new file; nothing is overwritten and nothing is refitted.

Usage:
    python retina_dr/rdr_predict.py site prepare --site deepdrid --kw tier1
    python retina_dr/rdr_predict.py site predict --site deepdrid --kw tier1 --members <tags>
    python retina_dr/rdr_predict.py idrid-views --kw idrid-views
    python retina_dr/rdr_predict.py boot --kw boot
    python retina_dr/rdr_predict.py viewstats --kw tier2 [--suffix t2cs]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import spearmanr

import rdr_common as P
from rdr_common import PRED_DIR, align_strict

RM = MC = MP = P
T2 = sys.modules[__name__]


# ==========================================================================================
# view runs
# ==========================================================================================

VIEWS = ('rgb', 'green', 'vessel', 'crop')
SEEDS = (0, 1, 2)
ORIGINAL_SUFFIX = 't2s'


def deep_tags(suffix=ORIGINAL_SUFFIX):
    """Run tags of the view runs per view: `t2s` geometric augmentation, `t2cs` with colour
    jitter and random-resized crop added (`rdr_train.py deep --aug colour`)."""
    return {v: tuple(f'retina-whole-{v}-{suffix}{s}' for s in SEEDS) for v in VIEWS}


def set_suffix(suffix):
    """Rebuild the module-level tag tables for `suffix`."""
    global SUFFIX, DEEP_T2, ALL_T2
    SUFFIX = suffix
    DEEP_T2 = deep_tags(suffix)
    ALL_T2 = tuple(t for v in VIEWS for t in DEEP_T2[v])


set_suffix(ORIGINAL_SUFFIX)
EXTRA = 'deepdrid'
VIEW_NEW_DIR = P.ARTIFACTS / 'newsite' / 'deepdrid'   # the anchor's DeepDRiD predictions
N_PERM = 100_000


def _rho(x, y):
    r = spearmanr(x, y)[0]
    return 0.0 if not np.isfinite(r) else float(r)


def f_stat(values, labels):
    values, labels = np.asarray(values, float), np.asarray(labels)
    groups = [values[labels == k] for k in np.unique(labels)]
    grand = values.mean()
    k, n = len(groups), len(values)
    between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    within = sum(((g - g.mean()) ** 2).sum() for g in groups) / (n - k)
    return float(between / within) if within > 0 else float('inf')


def f_perm_test(values, labels, n_perm=N_PERM, seed=MC.MECH_SEED):
    """One-way F with a permutation p: (1 + #{F_perm >= F_obs}) / (n_perm + 1)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    obs = f_stat(values, labels)
    hits = sum(f_stat(values, rng.permutation(labels)) >= obs - 1e-12 for _ in range(n_perm))
    return {'F': obs, 'p_perm': float((1 + hits) / (n_perm + 1)), 'n_perm': int(n_perm)}


def view_load(path, site):
    d = np.load(path, allow_pickle=False)
    return {'ids': d[f'{site}_ids'].astype(str), 'grade': d[f'{site}_grade'].astype(int),
            'proba5': d[f'{site}_proba5'].astype(np.float64)}


def view_cell(g, pa, pm):
    auc = lambda p: RM.cut_scores(g, p, MC.CUTS)['mean_auc']          # noqa: E731
    a, m = auc(pa), auc(pm)
    geo, ari = auc(RM.geo_blend(pm, pa, MC.W)), auc(RM.arith_blend(pm, pa, MC.W))
    return {'anchor': a, 'member': m, 'geo_contribution': geo - a,
            'arith_contribution': ari - a, 'geo_minus_arith': geo - ari}


def observe(m):
    per = {}
    for site in MC.OOD:
        _, g, pa, pm = MC.aligned(m, site)
        per[site] = view_cell(g, pa, pm)
    _, g, pa, pm, counts = align_strict(view_load(VIEW_NEW_DIR / f'pred_{MC.ANCHOR}.npz', EXTRA),
                                        view_load(PRED_DIR / f'pred_{m}.npz', EXTRA))
    per[EXTRA] = dict(view_cell(g, pa, pm), n=counts['n_aligned'])
    mean = lambda k: float(np.mean([per[s][k] for s in MC.OOD]))       # noqa: E731
    return {'per_site': per, 'contrib_geo': mean('geo_contribution'),
            'gap': mean('geo_minus_arith'), 'contrib_geo_deepdrid': per[EXTRA]['geo_contribution']}


def run_viewstats(run_tag):
    out = MC.out_path('views', run_tag)
    rows = {}
    for m in ALL_T2:
        _, g, pa, pm = MC.aligned(m, MC.ID_SITE)
        cuts = [MP.rescue_cut(g, pa, pm, c) for c in MC.CUTS]
        o = observe(m)
        rows[m] = {'view': m.split('-')[2], 'cuts': cuts,
                   'net_correction': float(np.mean([c['net'] for c in cuts])),
                   'ddr_test_mauc': RM.cut_scores(g, pm, MC.CUTS)['mean_auc'],
                   'contrib_geo': o['contrib_geo'],
                   'contrib_geo_deepdrid': o['contrib_geo_deepdrid'],
                   'gap': o['gap'], 'per_site': o['per_site']}
    labels = [rows[t]['view'] for t in ALL_T2]
    effect = {'am': f_perm_test([rows[t]['contrib_geo'] for t in ALL_T2], labels),
              'deepdrid': f_perm_test([rows[t]['contrib_geo_deepdrid'] for t in ALL_T2], labels),
              'ddr_auc': f_perm_test([rows[t]['ddr_test_mauc'] for t in ALL_T2], labels)}
    vmean = lambda k: [float(np.mean([rows[t][k] for t in DEEP_T2[v]])) for v in VIEWS]  # noqa: E731
    view_level = {'views': list(VIEWS),
                  'net_correction': vmean('net_correction'),
                  'contrib_geo': vmean('contrib_geo'),
                  'seed_sd_contrib_geo': [float(np.std([rows[t]['contrib_geo'] for t in DEEP_T2[v]],
                                                       ddof=1)) for v in VIEWS]}
    view_level['spearman'] = _rho(view_level['net_correction'], view_level['contrib_geo'])
    for m, r in rows.items():
        print(f'{m:26s} net={r["net_correction"]:+.4f} ddr mAUC={r["ddr_test_mauc"]:.4f} '
              f'contribution A/M={r["contrib_geo"]:+.4f} DeepDRiD={r["contrib_geo_deepdrid"]:+.4f}')
    for k, e in effect.items():
        print(f'view effect {k:8s} F={e["F"]:.3f} p={e["p_perm"]:.2g}')
    MC.write_json(out, {'schema': 'rdr_views/v1', 'run_tag': run_tag, 'w': MC.W, 'suffix': SUFFIX,
                        'inputs': MC.input_hashes((MC.ANCHOR,) + ALL_T2),
                        'view_effect': effect, 'view_level': view_level, 'rows': rows})


def main_viewstats(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--kw', required=True, help='run keyword (part of the run tag)')
    ap.add_argument('--suffix', default=ORIGINAL_SUFFIX,
                    help='view-run tag suffix: t2s (geometric augmentation) or t2cs (colour)')
    args = ap.parse_args(argv)
    set_suffix(args.suffix)
    run_viewstats(P.rdr_run_tag(args.kw))


# ==========================================================================================
# site: DeepDRiD and IDRiD manifests, caches, features and predictions
# ==========================================================================================
# DeepDRiD regular-fundus images and IDRiD disease grading, one site per invocation.
# Deep checkpoints and fitted classifiers are only loaded, never refitted.

SITE = 'deepdrid'            # set from --site in main_site()
ROOT = P.EXTERNAL / 'deepdrid' / 'repo' / 'regular_fundus_images'
FOLDS = (('regular-fundus-training', 'regular-fundus-training.csv'),
         ('regular-fundus-validation', 'regular-fundus-validation.csv'))
EVAL_FOLD = 'Online-Challenge1&2-Evaluation'
UNGRADABLE = 5
NEW_DIR = P.ARTIFACTS / 'newsite'
IDRID_ROOT = P.EXTERNAL / 'idrid' / 'extracted' / 'B. Disease Grading'
# Training and testing folds reuse the same file names (IDRiD_001 ...), so the image id
# carries the fold.
IDRID_FOLDS = (('train', 'a. Training Set', 'a. IDRiD_Disease Grading_Training Labels.csv'),
               ('test', 'b. Testing Set', 'b. IDRiD_Disease Grading_Testing Labels.csv'))
LESION_TAG = 'retina-lesion-les2'
DEEP_MEMBER = {MC.ANCHOR: 'patchbag', 'retina-whole-view': 'whole'}
MEMBERS_OVERRIDE = None      # set by --members; the candidate set, anchor fixed


def member_kind(tag):
    """'patchbag' / 'whole' / 'convnext' / 'dinov2' / 'effnet' for a deep run, 'feat' for a
    fitted classifier; from DEEP_MEMBER or the checkpoint."""
    if tag in DEEP_MEMBER:
        return DEEP_MEMBER[tag]
    pt = P.CKPT_DIR / f'{tag}.pt'
    if pt.is_file():
        import torch
        return torch.load(pt, map_location='cpu', weights_only=False)['member']
    if (P.CKPT_DIR / f'{tag}.joblib').is_file():
        return 'feat'
    raise RuntimeError(f'no checkpoint for {tag} under {P.CKPT_DIR}')


def _members():
    return tuple(MEMBERS_OVERRIDE) if MEMBERS_OVERRIDE else MC.MEMBERS


def _all_tags():
    return _members()


def _new_file(path):
    path = P.assert_rdr_output(path)
    if path.exists():
        raise SystemExit(f'refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _pred_path(tag):
    return NEW_DIR / SITE / f'pred_{tag}.npz'


# ------------------------------------------------------------------------------------------
# prepare
# ------------------------------------------------------------------------------------------

def _xlsx_rows(path):
    """First sheet of an .xlsx as lists of strings (no openpyxl in the env)."""
    z = zipfile.ZipFile(path)
    shared = []
    if 'xl/sharedStrings.xml' in z.namelist():
        x = z.read('xl/sharedStrings.xml').decode()
        shared = [re.sub(r'<[^>]+>', '', s) for s in re.findall(r'<si>(.*?)</si>', x, re.S)]
    x = z.read('xl/worksheets/sheet1.xml').decode()
    rows = []
    for r in re.findall(r'<row[^>]*>(.*?)</row>', x, re.S):
        vals = []
        for attrs, body in re.findall(r'<c([^>]*?)(?:/>|>(.*?)</c>)', r, re.S):
            v = re.search(r'<v>(.*?)</v>', body or '')
            v = v.group(1) if v else ''
            if 't="s"' in attrs and v:
                v = shared[int(v)]
            vals.append(v)
        rows.append(vals)
    return rows


def _image_path(fold, iid):
    return ROOT / fold / 'Images' / iid.split('_')[0] / f'{iid}.jpg'


def manifest_deepdrid():
    """Per-image grade from the eye the image shows. Dropped: grade 5, and an empty column for
    that eye (11 training images carry the grade in the other eye's column).
    -> (rows, n_raw, n_drop)."""
    raw = []
    for fold, name in FOLDS:
        with open(ROOT / fold / name) as fh:
            for rec in csv.DictReader(fh):
                iid = rec['image_id']
                eye = re.match(r'\d+_([lr])\d$', iid).group(1)
                g = rec['left_eye_DR_Level'] if eye == 'l' else rec['right_eye_DR_Level']
                raw.append((fold, iid, g.strip()))
    tab = _xlsx_rows(ROOT / EVAL_FOLD / 'Challenge1_labels.xlsx')
    if tab[0][:2] != ['image_id', 'DR_Levels']:
        raise RuntimeError(f'unexpected evaluation label header {tab[0]}')
    raw += [(EVAL_FOLD, r[0], r[1].strip()) for r in tab[1:] if r and r[0]]
    ids = [iid for _, iid, _ in raw]
    if len(set(ids)) != len(ids):
        raise RuntimeError('duplicate DeepDRiD image ids across folds')
    rows, n_drop = [], 0
    for fold, iid, g in raw:
        if g == '' or int(float(g)) == UNGRADABLE:
            n_drop += 1
            continue
        rows.append({'site': SITE, 'image_id': f'{iid}.jpg', 'path': str(_image_path(fold, iid)),
                     'grade': int(float(g)), 'split': 'ood'})
    return rows, len(raw), n_drop


def manifest_idrid():
    """IDRiD disease grading, both folds. It has no ungradable label. -> (rows, n_raw, n_drop)."""
    rows, n_raw, n_drop = [], 0, 0
    for fold, img_dir, label_csv in IDRID_FOLDS:
        with open(IDRID_ROOT / '2. Groundtruths' / label_csv, newline='') as fh:
            for rec in csv.DictReader(fh):
                name = (rec.get('Image name') or '').strip()
                if not name:
                    continue
                n_raw += 1
                g = rec['Retinopathy grade'].strip()
                if g == '' or int(g) == UNGRADABLE:
                    n_drop += 1
                    continue
                rows.append({'site': SITE, 'image_id': f'{fold}_{name}.jpg',
                             'path': str(IDRID_ROOT / '1. Original Images' / img_dir
                                         / f'{name}.jpg'),
                             'grade': int(g), 'split': 'ood'})
    return rows, n_raw, n_drop


BUILDERS = {'deepdrid': manifest_deepdrid, 'idrid': manifest_idrid}


def run_prepare(device_name, workers=16):
    import rdr_data as D
    import rdr_data as RF
    import rdr_data as PRE

    rows, n_raw, n_drop = BUILDERS[SITE]()
    missing = [r for r in rows if not P.Path(r['path']).is_file()]
    if missing:
        raise SystemExit(f'{len(missing)} manifest paths missing, e.g. {missing[0]["path"]}')
    hist = Counter(r['grade'] for r in rows)
    print(f'{SITE}: raw={n_raw} dropped={n_drop} kept={len(rows)} grades=' +
          ' '.join(f'{g}:{hist[g]}' for g in sorted(hist)), flush=True)
    out = P.assert_rdr_output(D.MANIFEST_DIR / f'{SITE}.csv')
    if out.exists():            # a re-run may reuse an identical manifest, never replace one
        old = [{**r, 'grade': int(r['grade'])} for r in D.read_manifest_rows(SITE)]
        if old != rows:
            raise SystemExit(f'{out} exists with different content')
        print(f'reusing identical {out}', flush=True)
    else:
        with open(out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, D.FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {out}', flush=True)

    payload = [(SITE, r['image_id'], r['path'], PRE.cache_path(SITE, r['image_id']))
               for r in rows]
    P.assert_rdr_output(P.CACHE_DIR / SITE)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(PRE._one, payload, chunksize=8))
    errs = [r for r in res if r[2].startswith('ERROR')]
    fr = [r[3] for r in res if r[3] is not None]
    print(f'cache: ok={sum(r[2] == "ok" for r in res)} cached={sum(r[2] == "cached" for r in res)}'
          f' errors={len(errs)} median crop fraction={np.median(fr) if fr else float("nan"):.3f}',
          flush=True)
    if errs:
        raise SystemExit(f'cache errors, e.g. {errs[:3]}')

    out = _new_file(P.ARTIFACTS / 'features' / f'{SITE}.npz')
    X = np.zeros((len(rows), len(RF.NAMES)), np.float32)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (_iid, f, err) in enumerate(ex.map(RF.extract_one,
                                                  [(SITE, r['image_id']) for r in rows],
                                                  chunksize=16)):
            if err:
                raise SystemExit(f'feature error on {_iid}: {err}')
            X[i] = f
    np.savez_compressed(out, X=X, ids=np.array([r['image_id'] for r in rows]),
                        grade=np.array([r['grade'] for r in rows]),
                        split=np.array([r['split'] for r in rows]), names=np.array(RF.NAMES))
    print(f'wrote {out} X={X.shape}', flush=True)

    import torch
    import rdr_train as LS
    out = _new_file(P.ARTIFACTS / 'features' / f'{SITE}_lesion.npz')
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    model = LS.build_unet().to(device)
    model.load_state_dict(torch.load(P.CKPT_DIR / f'{LESION_TAG}.pt', map_location='cpu',
                                     weights_only=False)['state_dict'])
    af = LS.area_fractions(model, [(SITE, r['image_id']) for r in rows], device)
    np.savez_compressed(out, X=af, ids=np.array([r['image_id'] for r in rows]),
                        names=np.array([f'lesion_{c}' for c in LS.CLASSES]))
    print(f'wrote {out} X={af.shape}', flush=True)


# ------------------------------------------------------------------------------------------
# predict
# ------------------------------------------------------------------------------------------

def _feat_matrix(names):
    """Site feature matrix in the column order a fitted classifier was trained on."""
    d = np.load(P.ARTIFACTS / 'features' / f'{SITE}.npz', allow_pickle=True)
    ids, X, cols = d['ids'].astype(str), d['X'], [str(n) for n in d['names']]
    dl = np.load(P.ARTIFACTS / 'features' / f'{SITE}_lesion.npz', allow_pickle=True)
    order = {v: i for i, v in enumerate(dl['ids'].astype(str))}
    X = np.concatenate([X, dl['X'][[order[v] for v in ids]]], axis=1)
    cols += [str(n) for n in dl['names']]
    idx = [cols.index(n) for n in names]
    return ids, d['grade'].astype(int), X[:, idx]


def run_predict_site(device_name):
    import joblib
    import torch

    import rdr_models as RModels
    import rdr_train as TD
    import rdr_train as TF

    outs = {t: _new_file(_pred_path(t))
            for t in (MC.ANCHOR,) + tuple(m for m in _all_tags() if m != MC.ANCHOR)}
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    rows = TD.read_rows(SITE, None)
    ids = np.array([r['image_id'] for r in rows])
    grade = np.array([int(r['grade']) for r in rows])
    summary = {}
    for tag, out in outs.items():
        kind = member_kind(tag)
        if kind != 'feat':
            ck = torch.load(P.CKPT_DIR / f'{tag}.pt', map_location='cpu', weights_only=False)
            a = ck['args']
            model = RModels.build(kind, pretrained=False, backbone=a.get('backbone')).to(device)
            model.load_state_dict(ck['state_dict'])
            bs = 16 if kind == 'patchbag' else 24
            p = TD.predict(model, rows, kind, device, int(a.get('k', 8)), bs)
            i_, g_ = ids, grade
            del model
            torch.cuda.empty_cache()
        else:
            ck = joblib.load(P.CKPT_DIR / f'{tag}.joblib')
            i_, g_, X = _feat_matrix(list(ck['names']))
            p = TF.proba5(ck['clf'], X)
        np.savez_compressed(out, **{f'{SITE}_ids': i_, f'{SITE}_grade': g_,
                                    f'{SITE}_proba5': p})
        summary[tag] = RM.cut_scores(g_, p, RM.RETINA_CUTS)['mean_auc']
        print(f'{tag:24s} n={len(i_)} mAUC={summary[tag]:.4f} -> {out}', flush=True)


def main_site(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('prepare', 'predict'))
    ap.add_argument('--site', required=True, choices=tuple(BUILDERS))
    ap.add_argument('--kw', required=True, help='run keyword (part of the run tag)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--members', default=None,
                    help='comma-separated candidate run tags (default: rdr_common.MEMBERS); anchor fixed')
    args = ap.parse_args(argv)
    global SITE, MEMBERS_OVERRIDE
    SITE = args.site
    MEMBERS_OVERRIDE = tuple(args.members.split(',')) if args.members else None
    if args.stage == 'prepare':
        run_prepare(args.device, args.workers)
    else:
        run_predict_site(args.device)


# ==========================================================================================
# idrid-views: the twelve view runs on IDRiD
# ==========================================================================================
# Writes mechanism/idrid_views/pred_<tag>.npz (keys idrid_ids/grade/proba5); skips existing
# files. The view runs score DeepDRiD during training (`--extra-site deepdrid`).

IDRID_SITE = 'idrid'
VIEW_DIR = MC.OUT_DIR / 'idrid_views'


def run_predict_idrid_views(device_name):
    import torch
    import rdr_models as RModels
    import rdr_train as TD

    P.assert_rdr_output(VIEW_DIR)
    VIEW_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    rows = TD.read_rows(IDRID_SITE, None)
    ids = np.array([r['image_id'] for r in rows])
    grade = np.array([int(r['grade']) for r in rows])
    for tag in ALL_T2:
        out = VIEW_DIR / f'pred_{tag}.npz'
        if out.exists():
            print(f'{tag:28s} exists, skip', flush=True)
            continue
        ck = torch.load(P.CKPT_DIR / f'{tag}.pt', map_location='cpu', weights_only=False)
        a = ck['args']
        assert a['member'] == 'whole', (tag, a['member'])
        model = RModels.build('whole', pretrained=False, backbone=a.get('backbone')).to(device)
        model.load_state_dict(ck['state_dict'])
        p = TD.predict(model, rows, 'whole', device, int(a.get('k', 8)), 24, view=a['view'])
        del model
        torch.cuda.empty_cache()
        np.savez_compressed(out, **{f'{IDRID_SITE}_ids': ids, f'{IDRID_SITE}_grade': grade,
                                    f'{IDRID_SITE}_proba5': p})
        print(f'{tag:28s} view={a["view"]:6s} n={len(ids)} mAUC='
              f'{RM.cut_scores(grade, p, RM.RETINA_CUTS)["mean_auc"]:.4f} -> {out}', flush=True)


def main_idrid_views(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--kw', required=True, help='run keyword (part of the run tag)')
    ap.add_argument('--device', default='cuda:0')
    run_predict_idrid_views(ap.parse_args(argv).device)


# ==========================================================================================
# boot: DDR bootstrap of net correction and mAUC for the view runs
# ==========================================================================================
# Resampling is stratified by ICDR grade, so both endpoints keep their class counts, and one
# draw is shared by every run and both cuts. A resample with replacement gives each image a
# count; a pair sum over the resample is then w_pos^T M w_neg for the pair matrix M, so no pair
# matrix is rebuilt. Unit weights reproduce `rdr_common.rescue_cut`.

B = 2000


def grade_weights(grade, rng):
    """[B, n] resample counts, stratified by grade."""
    w = np.zeros((B, grade.size))
    for k in np.unique(grade):
        idx = np.flatnonzero(grade == k)
        draws = rng.integers(0, idx.size, size=(B, idx.size))
        for b in range(B):
            w[b] += np.bincount(idx[draws[b]], minlength=grade.size)
    return w


def bilinear(wp, m, wn):
    """Row-wise wp[b] @ m @ wn[b] for a batch of weights."""
    return np.einsum('bj,bj->b', wp @ m, wn)


def cut_stats(g, pa, pm, c, wp_all, wn_all):
    """Point and bootstrap rescue, harm, net and member AUC for one cut."""
    pos, neg = MC.split_by_cut(g, c)
    sa, sm = MC.cut_score(pa, c), MC.cut_score(pm, c)
    ca = MC.pair_correct(sa[pos], sa[neg])
    cm = MC.pair_correct(sm[pos], sm[neg])
    a_wrong = 1.0 - ca
    mats = {'wrong': a_wrong, 'right': ca, 'rescued': a_wrong * cm,
            'harmed': ca * (1.0 - cm), 'member_correct': cm}
    point = {k: float(v.sum()) for k, v in mats.items()}
    wp, wn = wp_all[:, pos], wn_all[:, neg]
    boot = {k: bilinear(wp, v, wn) for k, v in mats.items()}
    total = wp.sum(axis=1) * wn.sum(axis=1)

    def derive(s, n_pairs):
        rescue = s['rescued'] / s['wrong']
        harm = s['harmed'] / s['right']
        return {'rescue_rate': rescue, 'harm_rate': harm, 'net': rescue - harm,
                'member_auc': s['member_correct'] / n_pairs}
    return derive(point, float(ca.size)), derive(boot, total)


def summary(x):
    x = np.asarray(x)
    lo, hi = np.percentile(x, [2.5, 97.5])
    return {'se': float(x.std(ddof=1)), 'ci95': [float(lo), float(hi)]}


def run_boot(run_tag):
    out = MC.out_path('ddr_boot', run_tag)
    out_npz = MC.out_path('ddr_boot', run_tag, 'npz')
    tags = ALL_T2
    ids0, g0, _, _ = MC.aligned(tags[0], MC.ID_SITE)
    rng = np.random.default_rng(MC.MECH_SEED)
    w_all = grade_weights(g0, rng)
    col = {i: k for k, i in enumerate(ids0)}
    rows, draws = {}, {}
    for t in tags:
        ids, g, pa, pm = MC.aligned(t, MC.ID_SITE)
        if ids.size != ids0.size:
            raise RuntimeError(f'{t}: {ids.size} aligned DDR ids, expected {ids0.size}')
        w = w_all[:, [col[i] for i in ids]]
        per_cut = {c: cut_stats(g, pa, pm, c, w, w) for c in MC.CUTS}
        net_b = np.mean([per_cut[c][1]['net'] for c in MC.CUTS], axis=0)
        mauc_b = np.mean([per_cut[c][1]['member_auc'] for c in MC.CUTS], axis=0)
        draws[t] = {'net': net_b, 'mauc': mauc_b}
        rows[t] = {'net_correction': {'point': float(np.mean([per_cut[c][0]['net'] for c in MC.CUTS])),
                                      **summary(net_b)},
                   'ddr_mauc': {'point': float(np.mean([per_cut[c][0]['member_auc'] for c in MC.CUTS])),
                                **summary(mauc_b)}}
        print(f'{t:28s} net={rows[t]["net_correction"]["point"]:.3f} '
              f'[{rows[t]["net_correction"]["ci95"][0]:.3f}, {rows[t]["net_correction"]["ci95"][1]:.3f}]')
    np.savez_compressed(out_npz, tags=np.array(tags),
                        **{f'{k}__{t}': v for t, d in draws.items() for k, v in d.items()})
    print(f'wrote {out_npz}')
    MC.write_json(out, {'schema': 'rdr_ddr_boot/v1', 'run_tag': run_tag, 'B': B,
                        'seed': MC.MECH_SEED, 'stratified_by': 'ddr grade',
                        'inputs': MC.input_hashes((MC.ANCHOR,) + tags), 'rows': rows,
                        'draws_npz': str(out_npz.name)})


def main_boot(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--kw', required=True, help='run keyword (part of the run tag)')
    run_boot(P.rdr_run_tag(ap.parse_args(argv).kw))


COMMANDS = {'site': main_site, 'idrid-views': main_idrid_views, 'boot': main_boot,
            'viewstats': main_viewstats}


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: {os.path.basename(sys.argv[0])} {'|'.join(COMMANDS)} [options]")
    COMMANDS[sys.argv[1]](sys.argv[2:])
