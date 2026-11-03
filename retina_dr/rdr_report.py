#!/usr/bin/env python
"""rdr_report.py -- every fundus number of the paper (Tables 1 and 2, Sections 3.1 to 3.3) for one
candidate set at a fixed combination weight.

Per candidate: DDR test mAUC alone; net correction on DDR test; for each operator (geometric,
arithmetic) at w: mAUC of anchor, candidate and combination on DDR test, APTOS, Messidor-2,
DeepDRiD and IDRiD; the worst-site change over APTOS and Messidor-2; the contribution
(combination minus anchor) on the A/M mean, DeepDRiD and IDRiD; paired image-level bootstraps of
the worst-site change (A/M) and of the IDRiD contribution. Over the candidates: exact Spearman
of net correction and of DDR mAUC with the geometric contribution per target.

Per view run: the same quantities; Spearman over the runs and over the view means; view-effect
one-way F with a permutation p; share of DDR bootstrap draws (`rdr_predict.py boot`) in which
rho(net correction) exceeds rho(mAUC).

Usage:
    python retina_dr/rdr_report.py --kw tuned --members <seven candidate tags in Table 1 order>
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import spearmanr

import rdr_common as P
import rdr_predict as T2
from rdr_common import PRED_DIR, align_strict

RM = MC = MP = P
OPS = {'geo': RM.geo_blend, 'arith': RM.arith_blend}
N_BOOT = 2000

NEWSITE = P.ARTIFACTS / 'newsite'
IDRID_VIEWS = MC.OUT_DIR / 'idrid_views'
EVAL_SITES = ('ddr', 'aptos', 'messidor2', 'deepdrid', 'idrid')
AM = ('aptos', 'messidor2')
TARGETS = {'am': 'contrib_am', 'deepdrid': 'contrib_deepdrid', 'idrid': 'contrib_idrid'}
VIEW_TAGS = tuple(t for v in T2.VIEWS for t in T2.DEEP_T2[v])


# ------------------------------------------------------------------------------------------
# bootstrap and permutation helpers
# ------------------------------------------------------------------------------------------

def _boot(g, pa, pm, n_boot=N_BOOT, seed=MC.MECH_SEED, w_geo=MC.W, w_arith=None):
    """Paired image-level bootstrap of geo and arith contribution (combination - anchor).
    `w_geo` / `w_arith`: the weight of each operator. The draws do not depend on w."""
    auc = lambda gg, p: RM.cut_scores(gg, p, MC.CUTS)['mean_auc']            # noqa: E731
    w_arith = w_geo if w_arith is None else w_arith
    geo, ari = RM.geo_blend(pm, pa, w_geo), RM.arith_blend(pm, pa, w_arith)
    point = {'anchor': auc(g, pa), 'member': auc(g, pm),
             'geo_contribution': auc(g, geo) - auc(g, pa),
             'arith_contribution': auc(g, ari) - auc(g, pa)}
    point['geo_minus_arith'] = point['geo_contribution'] - point['arith_contribution']
    rng = np.random.default_rng(seed)
    draws = {'geo_contribution': [], 'arith_contribution': [], 'geo_minus_arith': []}
    n = len(g)
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        gg = g[ix]
        if len(np.unique(gg >= MC.CUTS[0])) < 2 or len(np.unique(gg >= MC.CUTS[1])) < 2:
            continue
        a0 = auc(gg, pa[ix])
        dg, da = auc(gg, geo[ix]) - a0, auc(gg, ari[ix]) - a0
        draws['geo_contribution'].append(dg)
        draws['arith_contribution'].append(da)
        draws['geo_minus_arith'].append(dg - da)
    out = dict(point, n=int(n), n_boot_used=len(draws['geo_contribution']))
    for k, v in draws.items():
        v = np.asarray(v)
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[f'{k}_ci95'] = [float(lo), float(hi)]
        out[f'{k}_sd'] = float(v.std(ddof=1))
        out[f'{k}_p_gt0'] = float(np.mean(v > 0))
    return out


def _rho_perm(x, y, n_perm=100_000, seed=MC.MECH_SEED):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = float(spearmanr(x, y)[0])
    hits = sum(float(spearmanr(x, rng.permutation(y))[0]) >= obs - 1e-12 for _ in range(n_perm))
    return {'rho': obs, 'p_perm_one_sided': float((1 + hits) / (n_perm + 1)), 'n': int(len(x))}


def _boot_worst(sites, n_boot=N_BOOT, seed=MC.MECH_SEED, w_geo=MC.W, w_arith=None):
    """Paired bootstrap of Delta-worst = min_s mAUC(blend_s) - min_s mAUC(anchor_s) over the
    two original OOD sites; sites resampled independently, image-level, anchor and member
    rows paired. `sites` maps site -> (grade, p_anchor, p_member). `w_geo` / `w_arith` as in
    `_boot`."""
    auc = lambda gg, p: RM.cut_scores(gg, p, MC.CUTS)['mean_auc']            # noqa: E731
    w_arith = w_geo if w_arith is None else w_arith
    blends = {s: (RM.geo_blend(pm, pa, w_geo), RM.arith_blend(pm, pa, w_arith))
              for s, (g, pa, pm) in sites.items()}
    per_site = {s: {'anchor': auc(g, pa), 'geo': auc(g, blends[s][0]), 'arith': auc(g, blends[s][1])}
                for s, (g, pa, pm) in sites.items()}
    base = min(v['anchor'] for v in per_site.values())
    point = {'per_site': per_site, 'anchor_worst': base,
             'geo_d_worst': min(v['geo'] for v in per_site.values()) - base,
             'arith_d_worst': min(v['arith'] for v in per_site.values()) - base}
    point['geo_minus_arith'] = point['geo_d_worst'] - point['arith_d_worst']
    rng = np.random.default_rng(seed)
    draws = {'geo_d_worst': [], 'arith_d_worst': [], 'geo_minus_arith': []}
    for _ in range(n_boot):
        a0, ag, aa = [], [], []
        ok = True
        for s, (g, pa, pm) in sites.items():
            n = len(g)
            ix = rng.integers(0, n, n)
            gg = g[ix]
            if len(np.unique(gg >= MC.CUTS[0])) < 2 or len(np.unique(gg >= MC.CUTS[1])) < 2:
                ok = False
                break
            a0.append(auc(gg, pa[ix]))
            ag.append(auc(gg, blends[s][0][ix]))
            aa.append(auc(gg, blends[s][1][ix]))
        if not ok:
            continue
        dg, da = min(ag) - min(a0), min(aa) - min(a0)
        draws['geo_d_worst'].append(dg)
        draws['arith_d_worst'].append(da)
        draws['geo_minus_arith'].append(dg - da)
    out = dict(point, n={s: int(len(g)) for s, (g, _, _) in sites.items()},
               n_boot_used=len(draws['geo_d_worst']))
    for k, v in draws.items():
        v = np.asarray(v)
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[f'{k}_ci95'] = [float(lo), float(hi)]
        out[f'{k}_sd'] = float(v.std(ddof=1))
        out[f'{k}_p_gt0'] = float(np.mean(v > 0))
        out[f'{k}_p_lt0'] = float(np.mean(v < 0))
    return out


def ranks_desc(v):
    """1 = largest, along the last axis."""
    return (-v).argsort(axis=-1).argsort(axis=-1) + 1


def rho_rows(x, y):
    """Spearman of every row of x [B, n] with the fixed vector y (no ties in y)."""
    rx = ranks_desc(x).astype(float)
    ry = ranks_desc(np.asarray(y)[None, :]).astype(float)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean()
    return (rx * ry).sum(axis=1) / np.sqrt((rx ** 2).sum(axis=1) * (ry ** 2).sum())



# ------------------------------------------------------------------------------------------
# predictions
# ------------------------------------------------------------------------------------------

def pred_file(tag, site):
    """(path, key prefix) of `tag`'s predictions on `site`; first existing location wins."""
    cands = [(NEWSITE / site / f'pred_{tag}.npz', site), (PRED_DIR / f'pred_{tag}.npz', site)]
    if site == 'idrid':
        cands.append((IDRID_VIEWS / f'pred_{tag}.npz', site))
    for path, key in cands:
        if path.is_file() and f'{key}_ids' in np.load(path, allow_pickle=False).files:
            return path, key
    raise FileNotFoundError(f'no {site} predictions for {tag}')


def load(tag, site):
    path, key = pred_file(tag, site)
    d = np.load(path, allow_pickle=False)
    return {'ids': d[f'{key}_ids'].astype(str), 'grade': d[f'{key}_grade'].astype(int),
            'proba5': d[f'{key}_proba5'].astype(np.float64)}


def aligned(anchor, member, site):
    _, g, pa, pm, _ = align_strict(load(anchor, site), load(member, site))
    return g, pa, pm


def mauc(g, p):
    return RM.cut_scores(g, p, RM.RETINA_CUTS)['mean_auc']


# ------------------------------------------------------------------------------------------
# one member
# ------------------------------------------------------------------------------------------

def member_report(anchor, tag, w_fixed, n_boot, sites=EVAL_SITES):
    out = {}
    gt, pat, pmt = aligned(anchor, tag, 'ddr')
    out['ddr_test_mauc'] = mauc(gt, pmt)
    cuts = [MP.rescue_cut(gt, pat, pmt, c) for c in MC.CUTS]
    out['net_correction'] = float(np.mean([c['net'] for c in cuts]))
    out['net_cuts'] = cuts
    evals = {s: aligned(anchor, tag, s) for s in sites}
    out['alone'] = {s: mauc(g, pm) for s, (g, _, pm) in evals.items()}
    out['anchor'] = {s: mauc(g, pa) for s, (g, pa, _) in evals.items()}
    out['alone_worst_am'] = min(out['alone'][s] for s in AM)
    anchor_worst = min(out['anchor'][s] for s in AM)

    out['modes'] = {}
    for mode in ('fixed',):
        m = out['modes'][mode] = {}
        for op, fn in OPS.items():
            w = w_fixed
            combo = {s: mauc(g, fn(pm, pa, w)) for s, (g, pa, pm) in evals.items()}
            r = m[op] = {'w': w, 'combo': combo,
                         'contrib': {s: combo[s] - out['anchor'][s] for s in sites},
                         'd_worst_am': min(combo[s] for s in AM) - anchor_worst,
                         'combo_worst_am': min(combo[s] for s in AM)}
            r['contrib_am'] = float(np.mean([r['contrib'][s] for s in AM]))
            r['contrib_deepdrid'] = r['contrib'].get('deepdrid')
            r['contrib_idrid'] = r['contrib'].get('idrid')
            r['combo_below_alone_worst_am'] = bool(r['combo_worst_am'] < out['alone_worst_am'])
        m['geo_minus_arith_d_worst'] = m['geo']['d_worst_am'] - m['arith']['d_worst_am']
        if n_boot:
            wg, wa = m['geo']['w'], m['arith']['w']
            m['boot_am'] = _boot_worst({s: evals[s] for s in AM}, n_boot=n_boot,
                                       w_geo=wg, w_arith=wa)
            if 'idrid' in evals:
                g, pa, pm = evals['idrid']
                m['boot_idrid'] = _boot(g, pa, pm, n_boot=n_boot, w_geo=wg, w_arith=wa)
    return out


# ------------------------------------------------------------------------------------------
# over members
# ------------------------------------------------------------------------------------------

def candidate_stats(reports, names):
    """Exact Spearman (n <= 8) of net correction and DDR mAUC with the geometric
    contribution, per target; and whether the two rank identically."""
    out = {}
    net = [reports[m]['net_correction'] for m in names]
    ma = [reports[m]['ddr_test_mauc'] for m in names]
    out['net_and_mauc_same_rank'] = bool(np.array_equal(np.argsort(net), np.argsort(ma)))
    for mode in ('fixed',):
        out[mode] = {}
        for tgt, key in TARGETS.items():
            y = [reports[m]['modes'][mode]['geo'][key] for m in names]
            if any(v is None for v in y):
                continue
            out[mode][tgt] = {
                'net': {alt: MC.spearman_exact(net, y, alternative=alt)
                        for alt in ('two-sided', 'greater')},
                'mauc': {alt: MC.spearman_exact(ma, y, alternative=alt)
                         for alt in ('two-sided', 'greater')},
                'n_geo_contrib_positive': int(sum(v > 0 for v in y)),
                'n_geo_minus_arith_nonneg': int(sum(
                    reports[m]['modes'][mode]['geo'][key]
                    - reports[m]['modes'][mode]['arith'][key] >= 0 for m in names))}
    return out


def view_stats(reports, boot_run):
    """Spearman over the 12 runs and the 4 view means, view-effect F, DDR-bootstrap share."""
    view_of = {t: v for v in T2.VIEWS for t in T2.DEEP_T2[v]}
    net = [reports[t]['net_correction'] for t in VIEW_TAGS]
    ma = [reports[t]['ddr_test_mauc'] for t in VIEW_TAGS]
    labels = [view_of[t] for t in VIEW_TAGS]
    npz = np.load(MC.OUT_DIR / f'ddr_boot_{boot_run}.npz')
    draw = {k: np.stack([npz[f'{k}__{t}'] for t in VIEW_TAGS], axis=1) for k in ('net', 'mauc')}
    out = {'view_means': {}, 'n_runs': len(VIEW_TAGS)}
    for mode in ('fixed',):
        out[mode] = {}
        out['view_means'][mode] = {
            v: {'net': float(np.mean([reports[t]['net_correction'] for t in T2.DEEP_T2[v]])),
                'mauc': float(np.mean([reports[t]['ddr_test_mauc'] for t in T2.DEEP_T2[v]])),
                'w_geo': [reports[t]['modes'][mode]['geo']['w'] for t in T2.DEEP_T2[v]],
                **{key: float(np.mean([reports[t]['modes'][mode]['geo'][key]
                                       for t in T2.DEEP_T2[v]])) for key in TARGETS.values()}}
            for v in T2.VIEWS}
        for tgt, key in TARGETS.items():
            y = [reports[t]['modes'][mode]['geo'][key] for t in VIEW_TAGS]
            vm = out['view_means'][mode]
            d = rho_rows(draw['net'], y) - rho_rows(draw['mauc'], y)
            out[mode][tgt] = {
                'rho_net': float(spearmanr(net, y)[0]), 'rho_mauc': float(spearmanr(ma, y)[0]),
                'rho_net_perm': _rho_perm(net, y), 'rho_mauc_perm': _rho_perm(ma, y),
                'view_means_rho_net': float(spearmanr([vm[v]['net'] for v in T2.VIEWS],
                                                      [vm[v][key] for v in T2.VIEWS])[0]),
                'view_means_rho_mauc': float(spearmanr([vm[v]['mauc'] for v in T2.VIEWS],
                                                       [vm[v][key] for v in T2.VIEWS])[0]),
                'view_effect': T2.f_perm_test(y, labels),
                'ddr_boot_rho_net_minus_mauc': {
                    'median': float(np.median(d)),
                    'ci95': [float(x) for x in np.percentile(d, [2.5, 97.5])],
                    'p_positive': float((d > 0).mean()), 'n_draws': int(len(d))}}
    return out


def print_tables(rep, names, cs, vs):
    for mode in ('fixed',):
        print(f'\n== candidates, w = {rep[names[0]]["modes"][mode]["geo"]["w"]} ==')
        print(f'{"member":26s} {"ID":5s} {"net":5s} | {"wG":4s} {"dW_G":6s} {"wA":4s} {"dW_A":6s} | '
              f'{"A/M":6s} {"DeepDR":6s} {"IDRiD":6s} | {"alone":5s} {"combo":5s}')
        for m in names:
            r, g, a = rep[m], rep[m]['modes'][mode]['geo'], rep[m]['modes'][mode]['arith']
            print(f'{m:26s} {r["ddr_test_mauc"]:.3f} {r["net_correction"]:.2f} | '
                  f'{g["w"]:.2f} {g["d_worst_am"]:+.3f} {a["w"]:.2f} {a["d_worst_am"]:+.3f} | '
                  f'{g["contrib_am"]:+.3f} {(g["contrib_deepdrid"] or 0):+.3f} '
                  f'{(g["contrib_idrid"] or 0):+.3f} | {r["alone_worst_am"]:.3f} {g["combo_worst_am"]:.3f}')
        for tgt, d in cs[mode].items():
            print(f'  rho {tgt:8s} net {d["net"]["two-sided"]["rho"]:+.3f} '
                  f'(p2 {d["net"]["two-sided"]["p_exact"]:.4f}, p1 {d["net"]["greater"]["p_exact"]:.4f}) '
                  f'mAUC {d["mauc"]["two-sided"]["rho"]:+.3f}; geo>0 {d["n_geo_contrib_positive"]}/{len(names)}; '
                  f'geo>=arith {d["n_geo_minus_arith_nonneg"]}/{len(names)}')
        if vs:
            print(f'  views ({mode}):')
            for tgt, d in vs[mode].items():
                print(f'    {tgt:8s} rho net {d["rho_net"]:+.3f} mAUC {d["rho_mauc"]:+.3f} | '
                      f'view means net {d["view_means_rho_net"]:+.2f} mAUC {d["view_means_rho_mauc"]:+.2f} | '
                      f'F {d["view_effect"]["F"]:.1f} p {d["view_effect"]["p_perm"]:.2g} | '
                      f'P(rho net > rho mAUC) {d["ddr_boot_rho_net_minus_mauc"]["p_positive"]:.4f}')
            for v, d in vs['view_means'][mode].items():
                print(f'    {v:7s} w {d["w_geo"]} net {d["net"]:.3f} mAUC {d["mauc"]:.4f} '
                      f'A/M {d["contrib_am"]:+.4f} DeepDRiD {d["contrib_deepdrid"]:+.4f} IDRiD {d["contrib_idrid"]:+.4f}')
    print('\nnet correction and DDR mAUC rank the candidates identically:', cs['net_and_mauc_same_rank'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kw', required=True, help='run keyword (part of the run tag)')
    ap.add_argument('--anchor', default=MC.ANCHOR)
    ap.add_argument('--members', default=','.join(MC.MEMBERS), help='candidate run tags, Table 1 order')
    ap.add_argument('--w-fixed', type=float, default=MC.W)
    ap.add_argument('--n-boot', type=int, default=2000, help='0 skips the bootstraps')
    ap.add_argument('--boot', default='retina-boot', help='run tag of `rdr_predict.py boot` (view runs)')
    ap.add_argument('--no-views', action='store_true')
    args = ap.parse_args()
    run_tag = P.rdr_run_tag(args.kw)
    out = MC.out_path('tuned_report', run_tag)
    names = args.members.split(',')

    rep = {}
    for m in names:
        rep[m] = member_report(args.anchor, m, args.w_fixed, args.n_boot)
        g = rep[m]['modes']['fixed']['geo']
        print(f'{m:26s} ID {rep[m]["ddr_test_mauc"]:.4f} net {rep[m]["net_correction"]:+.3f} '
              f'geo w {g["w"]:.2f} dW {g["d_worst_am"]:+.4f}', flush=True)
    cs = candidate_stats(rep, names)
    vrep, vs = {}, None
    if not args.no_views:
        for t in VIEW_TAGS:
            vrep[t] = member_report(args.anchor, t, args.w_fixed, args.n_boot,
                                    sites=('ddr', 'aptos', 'messidor2', 'deepdrid', 'idrid'))
            print(f'{t:26s} view run done', flush=True)
        vs = view_stats(vrep, args.boot)
    print_tables(rep, names, cs, vs)

    MC.write_json(out, {
        'schema': 'rdr_tuned_report/v1', 'run_tag': run_tag, 'anchor': args.anchor,
        'members': names, 'w_fixed': args.w_fixed, 'n_boot': args.n_boot,
        'inputs': {t: MC.sha256(pred_file(t, 'ddr')[0]) for t in (args.anchor,) + tuple(names)},
        'candidates': rep, 'candidate_stats': cs,
        'view_runs': vrep, 'view_stats': vs, 'view_tags': list(VIEW_TAGS)})
    print('wrote', out)


if __name__ == '__main__':
    main()
