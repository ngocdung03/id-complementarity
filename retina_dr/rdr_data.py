"""Site manifests and data gate, preprocessing, handcrafted features and the controlled input views.

Subcommands: manifests, preproc, features, views.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np


# ==========================================================================================
# rdr_data.py
# ==========================================================================================
# rdr_data.py -- site manifests for the retinal DR arm.
#
# One manifest CSV per site with columns {site, image_id, path, grade, split}, written to
# artifacts/retina_dr/manifests/. Nothing here reads pixels; it resolves labels and paths and
# runs the data gate (published counts, duplicate check, per-site grade histogram).
#
# Label conventions:
#   * ICDR grades 0-4. DDR additionally uses 5 for UNGRADABLE -> dropped.
#   * Messidor-2 rows with adjudicated_gradable == 0 -> dropped.
#   * APTOS: only the competition `train` split carries labels, so it is the only part used.
#
# De-duplication: DDR's official lists ship byte-identical images under different filenames
# across splits (6 train/valid, 3 valid/test as of 9 October 2026; no train/test pair). Each
# such image is kept only in its earliest split, so no evaluated image was ever trained on. A
# train/test collision is treated as fatal rather than handled.
#
# Splits: DDR carries the official train / valid / test lists and is the ID site. Every other
# site is held out entirely and its rows are labelled split='ood' -- they never touch training,
# checkpointing, preprocessing parameters or thresholds.
#
# Usage:
#     python retina_dr/rdr_data.py manifests            # build manifests + print the gate
#     python retina_dr/rdr_data.py manifests --gate     # gate only, no writes
#     python retina_dr/rdr_data.py manifests --rehash   # ignore the md5 cache

import rdr_common as P

MANIFEST_DIR = P.ARTIFACTS / 'manifests'

# Published counts, for the gate. DDR is quoted inclusive of its ungradable class.
PUBLISHED = {'ddr': 13673, 'aptos': 3662, 'messidor2': 1748}

UNGRADABLE_DDR = 5
FIELDS = ('site', 'image_id', 'path', 'grade', 'split')


def _row(site, image_id, path, grade, split):
    return {'site': site, 'image_id': image_id, 'path': str(path),
            'grade': int(grade), 'split': split}


def manifest_ddr():
    """DDR official DR_grading lists. Returns (rows, n_raw, n_ungradable)."""
    rows, n_raw, n_ung = [], 0, 0
    for split in ('train', 'valid', 'test'):
        listing = P.DDR_GRADING / f'{split}.txt'
        img_dir = P.DDR_GRADING / split
        with open(listing) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                name, grade = line.rsplit(None, 1)
                n_raw += 1
                if int(grade) == UNGRADABLE_DDR:
                    n_ung += 1
                    continue
                rows.append(_row('ddr', name, img_dir / name, grade, split))
    return rows, n_raw, n_ung


def manifest_aptos():
    """APTOS 2019; only the labelled competition train split exists."""
    rows, n_raw = [], 0
    with open(P.APTOS_ROOT / 'train.csv') as fh:
        for rec in csv.DictReader(fh):
            n_raw += 1
            rows.append(_row('aptos', rec['id_code'] + '.png',
                             P.APTOS_ROOT / 'train_images' / (rec['id_code'] + '.png'),
                             rec['diagnosis'], 'ood'))
    return rows, n_raw, 0


def manifest_messidor2():
    """Messidor-2 adjudicated grades (Google Brain release); ungradable rows dropped."""
    rows, n_raw, n_ung = [], 0, 0
    with open(P.MESSIDOR_ROOT / 'messidor_data.csv') as fh:
        for rec in csv.DictReader(fh):
            n_raw += 1
            grade = rec['adjudicated_dr_grade'].strip()
            if rec['adjudicated_gradable'].strip() != '1' or grade == '':
                n_ung += 1
                continue
            rows.append(_row('messidor2', rec['image_id'],
                             P.MESSIDOR_ROOT / 'messidor-2' / 'images' / rec['image_id'],
                             float(grade), 'ood'))
    return rows, n_raw, n_ung


def manifest_idrid():
    """IDRiD -- manual IEEE Dataport download, not present as of 9 October 2026."""
    return [], 0, 0


BUILDERS = {'ddr': manifest_ddr, 'aptos': manifest_aptos,
            'messidor2': manifest_messidor2, 'idrid': manifest_idrid}


def read_manifest_rows(site):
    """Read back a written manifest. The single reader used by training and analysis, so a
    site's rows cannot be assembled two different ways."""
    path = MANIFEST_DIR / f'{site}.csv'
    if not path.is_file():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(chunk), b''):
            h.update(blk)
    return h.hexdigest()


MD5_CACHE = MANIFEST_DIR / 'ddr_md5.csv'

# Split precedence for content de-duplication. An image whose bytes already appeared in an
# EARLIER split is dropped from the later one, so training keeps its copy and no evaluated
# image was ever trained on. DDR's official lists do contain such pairs (verified 9 October
# 2026: 6 train/valid and 3 valid/test, none train/test).
SPLIT_RANK = {'train': 0, 'valid': 1, 'test': 2}


def ddr_md5s(rows, use_cache=True):
    """md5 per DDR image, cached -- hashing 12.5k JPEGs is minutes, and the manifest is a
    deliverable in its own right."""
    want = {r['image_id']: r['path'] for r in rows if r['site'] == 'ddr'}
    cached = {}
    if use_cache and MD5_CACHE.is_file():
        with open(MD5_CACHE) as fh:
            cached = {rec['image_id']: rec['md5'] for rec in csv.DictReader(fh)}
    out = {k: cached[k] for k in want if k in cached}
    todo = [k for k in want if k not in out]
    for k in todo:
        out[k] = md5(want[k])
    if todo:
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        with open(P.assert_rdr_output(MD5_CACHE), 'w', newline='') as fh:
            w = csv.DictWriter(fh, ('image_id', 'md5'))
            w.writeheader()
            w.writerows({'image_id': k, 'md5': v} for k, v in sorted(out.items()))
    return out, len(todo)


def duplicate_check(rows, digests=None):
    """Byte-identical images appearing in more than one DDR split.

    Checked on content, not filename: the pairs DDR actually ships have different filenames,
    so a name-based check sees nothing. A train/test pair would void every ID number.
    """
    ddr = [r for r in rows if r['site'] == 'ddr']
    if digests is None:
        digests = {r['image_id']: md5(r['path']) for r in ddr}
    by_digest = {}
    for r in sorted(ddr, key=lambda r: (SPLIT_RANK[r['split']], r['image_id'])):
        by_digest.setdefault(digests[r['image_id']], []).append(r)
    dupes = [grp for grp in by_digest.values()
             if len({r['split'] for r in grp}) > 1]
    return dupes


def drop_cross_split_duplicates(rows, dupes):
    """Keep each duplicated image only in its earliest split; return (kept_rows, dropped)."""
    drop_ids = set()
    for grp in dupes:
        for r in grp[1:]:
            drop_ids.add((r['site'], r['image_id'], r['split']))
    kept = [r for r in rows if (r['site'], r['image_id'], r['split']) not in drop_ids]
    return kept, drop_ids


def main_manifests(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', action='store_true', help='gate only, write nothing')
    ap.add_argument('--skip-md5', action='store_true',
                    help='skip the cross-split duplicate check (first run hashes 12.5k JPEGs)')
    ap.add_argument('--rehash', action='store_true',
                    help='ignore the md5 cache and re-hash every DDR image')
    args = ap.parse_args(argv)

    all_rows, failures = [], []
    for site in P.SITES:
        rows, n_raw, n_dropped = BUILDERS[site]()
        if not rows:
            print(f'{site:10s} ABSENT -- skipped')
            if site != 'idrid':
                failures.append(f'{site} produced no rows')
            continue
        missing = [r for r in rows if not P.Path(r['path']).is_file()]
        hist = Counter(r['grade'] for r in rows)
        print(f'{site:10s} raw={n_raw:<6d} dropped={n_dropped:<5d} kept={len(rows):<6d} '
              f'missing_files={len(missing):<4d} grades=' +
              ' '.join(f'{g}:{hist[g]}' for g in sorted(hist)))
        for split in sorted({r['split'] for r in rows}):
            sub = [r for r in rows if r['split'] == split]
            sh = Counter(r['grade'] for r in sub)
            prev2 = sum(v for g, v in sh.items() if g >= 2) / len(sub)
            prev3 = sum(v for g, v in sh.items() if g >= 3) / len(sub)
            print(f'{"":12s}{split:6s} n={len(sub):<6d} '
                  f'prev(>=2)={prev2:.3f} prev(>=3)={prev3:.3f}')
        if site in PUBLISHED and n_raw != PUBLISHED[site]:
            failures.append(f'{site}: raw count {n_raw} != published {PUBLISHED[site]}')
        if missing:
            failures.append(f'{site}: {len(missing)} manifest paths do not exist')
        all_rows.extend(rows)

    dropped = set()
    if not args.skip_md5:
        digests, n_hashed = ddr_md5s(all_rows, use_cache=not args.rehash)
        dupes = duplicate_check(all_rows, digests)
        n_pairs = sum(len(g) - 1 for g in dupes)
        print(f'{"ddr md5":10s} hashed_now={n_hashed} cached={len(digests) - n_hashed} '
              f'cross-split duplicate groups={len(dupes)} redundant_copies={n_pairs}')
        for grp in dupes:
            keep, rest = grp[0], grp[1:]
            print(f'{"":12s}keep {keep["split"]:5s} {keep["image_id"]:24s} drop ' +
                  ', '.join(f'{r["split"]}/{r["image_id"]}' for r in rest))
        # A train/test collision would void every ID number; that one is fatal, not handled.
        for grp in dupes:
            splits = {r['split'] for r in grp}
            if 'train' in splits and 'test' in splits:
                failures.append(f'ddr: train/test content collision on {grp[0]["image_id"]}')
        all_rows, dropped = drop_cross_split_duplicates(all_rows, dupes)
        if dropped:
            per = Counter(split for _, _, split in dropped)
            print(f'{"":12s}dropped from evaluation splits: ' +
                  ' '.join(f'{k}:{v}' for k, v in sorted(per.items())))

    # The per-site block above reports PRE-dedup counts, because the duplicate check needs
    # every row. Restate the final composition so the printed gate matches the manifests.
    print('\nfinal manifest composition (post de-duplication):')
    for site in sorted({r['site'] for r in all_rows}):
        sub = [r for r in all_rows if r['site'] == site]
        per = Counter(r['split'] for r in sub)
        print(f'{site:10s} n={len(sub):<6d} ' + ' '.join(f'{k}:{v}' for k, v in sorted(per.items())))

    if not args.gate:
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        P.assert_rdr_output(MANIFEST_DIR)
        for site in sorted({r['site'] for r in all_rows}):
            out = MANIFEST_DIR / f'{site}.csv'
            with open(P.assert_rdr_output(out), 'w', newline='') as fh:
                w = csv.DictWriter(fh, FIELDS)
                w.writeheader()
                w.writerows(r for r in all_rows if r['site'] == site)
            print(f'wrote {out}')

    if failures:
        print('\nGATE FAIL')
        for f in failures:
            print(' -', f)
        sys.exit(1)
    print('\nGATE PASS')


# ==========================================================================================
# rdr_preproc.py
# ==========================================================================================
# rdr_preproc.py -- one preprocessing path for every site, plus the DDR lesion masks.
#
# Fundus photographs arrive with the retina as a bright disc on a black surround, and the
# surround's size differs by camera and by site. Left as-is it is a site cue a model can key
# on, so every image gets the same treatment: crop to the fundus disc, then resize.
#
# Cached at 1024 px long side as JPEG quality 95. The 512 px whole-image view is derived by
# downscaling at load time rather than cached separately: a resize is negligible beside a
# forward pass, and it halves the cache. Nothing here depends on a site label or on train/test
# membership.
#
# Usage:
#     python retina_dr/rdr_data.py preproc --site all --workers 16
#     python retina_dr/rdr_data.py preproc --lesions                # DDR EX/HE/MA/SE masks
#     python retina_dr/rdr_data.py preproc --smoke 8                # crop sanity on a few images per site

import rdr_common as P

LONG_SIDE = 1024
JPEG_Q = 95
# Fraction of the per-image max intensity below which a pixel counts as surround. Fixed for
# every site; 0.05 keeps dim but genuine retina while rejecting sensor noise in the black
# border. Verified by --smoke across all three sites.
BG_FRAC = 0.05
LESION_CLASSES = ('EX', 'HE', 'MA', 'SE')


def fundus_bbox(bgr, bg_frac=BG_FRAC):
    """Tight bounding box of the fundus disc. Returns (y0, y1, x0, x1)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thr = max(1.0, float(gray.max()) * bg_frac)
    mask = gray > thr
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:          # all-black read; keep the frame
        return 0, bgr.shape[0], 0, bgr.shape[1]
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def resize_long(img, long_side=LONG_SIDE, interp=cv2.INTER_AREA):
    h, w = img.shape[:2]
    s = long_side / float(max(h, w))
    if s >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                      interpolation=interp)


def preprocess_image(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f'unreadable: {path}')
    y0, y1, x0, x1 = fundus_bbox(bgr)
    return resize_long(bgr[y0:y1, x0:x1]), (y0, y1, x0, x1), bgr.shape[:2]


def _one(args):
    site, image_id, src, dst = args
    try:
        if os.path.exists(dst):
            return (site, image_id, 'cached', None)
        img, bbox, orig = preprocess_image(src)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not cv2.imwrite(dst, img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q]):
            raise RuntimeError('imwrite failed')
        frac = ((bbox[1] - bbox[0]) * (bbox[3] - bbox[2])) / float(orig[0] * orig[1])
        return (site, image_id, 'ok', frac)
    except Exception as exc:                                        # noqa: BLE001
        return (site, image_id, f'ERROR {exc}', None)


def cache_path(site, image_id):
    return str(P.CACHE_DIR / site / (os.path.splitext(image_id)[0] + '.jpg'))


def read_manifest(site):
    path = P.ARTIFACTS / 'manifests' / f'{site}.csv'
    if not path.is_file():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def build_site_cache(site, workers):
    rows = read_manifest(site)
    if not rows:
        print(f'{site}: no manifest, skipped')
        return
    payload = [(site, r['image_id'], r['path'], cache_path(site, r['image_id'])) for r in rows]
    P.assert_rdr_output(P.CACHE_DIR / site)
    ok = cached = 0
    errs, fracs = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (s, iid, status, frac) in enumerate(ex.map(_one, payload, chunksize=16), 1):
            if status == 'ok':
                ok += 1
                fracs.append(frac)
            elif status == 'cached':
                cached += 1
            else:
                errs.append((iid, status))
            if i % 2000 == 0:
                print(f'  {site} {i}/{len(payload)}', flush=True)
    msg = f'{site:10s} written={ok} already_cached={cached} errors={len(errs)}'
    if fracs:
        msg += (f'  kept_area median={np.median(fracs):.3f} '
                f'min={np.min(fracs):.3f} max={np.max(fracs):.3f}')
    print(msg, flush=True)
    for iid, st in errs[:10]:
        print(f'    {iid}: {st}', flush=True)


def ddr_lesion_dirs():
    """DDR's valid split names its label folder 'segmentation label', the others 'label'."""
    out = {}
    for split in ('train', 'valid', 'test'):
        base = P.DDR_LESION / split
        lab = base / 'label'
        if not lab.is_dir():
            lab = base / 'segmentation label'
        out[split] = (base / 'image', lab)
    return out


def build_lesions(workers):
    """Cache DDR lesion masks through the SAME crop and resize as their image.

    Region masking only means anything if mask and image are in register, so the bbox is
    taken from the image and applied to the mask -- never recomputed on the mask, which is
    mostly empty and would crop to the lesions themselves.
    """
    jobs = []
    for split, (img_dir, lab_dir) in ddr_lesion_dirs().items():
        if not img_dir.is_dir():
            continue
        for name in sorted(os.listdir(img_dir)):
            stem = os.path.splitext(name)[0]
            jobs.append((split, stem, str(img_dir / name), str(lab_dir)))
    print(f'lesion images: {len(jobs)}')
    done = fail = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for stem, status in ex.map(_one_lesion, jobs, chunksize=4):
            if status == 'ok':
                done += 1
            else:
                fail += 1
                if fail <= 10:
                    print(f'    {stem}: {status}')
    print(f'lesion masks cached for {done} images, {fail} failed')


def _one_lesion(args):
    split, stem, img_path, lab_dir = args
    try:
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError('unreadable image')
        y0, y1, x0, x1 = fundus_bbox(bgr)
        target = resize_long(bgr[y0:y1, x0:x1]).shape[:2]
        out_dir = P.CACHE_DIR / 'ddr_lesion' / split
        for cls in LESION_CLASSES:
            src = None
            for ext in ('.tif', '.png', '.jpg'):
                cand = os.path.join(lab_dir, cls, stem + ext)
                if os.path.exists(cand):
                    src = cand
                    break
            if src is None:
                raise RuntimeError(f'missing {cls} mask')
            m = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise RuntimeError(f'unreadable {cls} mask')
            m = (m[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
            m = cv2.resize(m, (target[1], target[0]), interpolation=cv2.INTER_NEAREST)
            d = out_dir / cls
            os.makedirs(d, exist_ok=True)
            if not cv2.imwrite(str(d / (stem + '.png')), m):
                raise RuntimeError(f'imwrite {cls}')
        return stem, 'ok'
    except Exception as exc:                                        # noqa: BLE001
        return stem, f'ERROR {exc}'


def smoke(n):
    """Crop sanity per site: how much of the frame survives, and does it stay square-ish."""
    for site in ('ddr', 'aptos', 'messidor2'):
        rows = read_manifest(site)[:n]
        if not rows:
            continue
        print(f'--- {site}')
        for r in rows:
            img, bbox, orig = preprocess_image(r['path'])
            kept = ((bbox[1] - bbox[0]) * (bbox[3] - bbox[2])) / float(orig[0] * orig[1])
            print(f"  {r['image_id'][:28]:30s} orig={orig} crop={img.shape[:2]} "
                  f"kept={kept:.3f} aspect={img.shape[1] / img.shape[0]:.3f}")


def main_preproc(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', default='all', choices=('all', 'ddr', 'aptos', 'messidor2'))
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--lesions', action='store_true')
    ap.add_argument('--smoke', type=int, default=0)
    args = ap.parse_args(argv)

    if args.smoke:
        smoke(args.smoke)
        return
    if args.lesions:
        build_lesions(args.workers)
        return
    sites = ('ddr', 'aptos', 'messidor2') if args.site == 'all' else (args.site,)
    for s in sites:
        build_site_cache(s, args.workers)


# ==========================================================================================
# rdr_features.py
# ==========================================================================================
# rdr_features.py -- the handcrafted retina view, and a measured check on its lesion proxies.
#
# Feature families (28 dims), all computed on the cached fundus crop:
#   colour   per-channel mean, std, and 10/50/90th percentiles                     15
#   texture  GLCM contrast / homogeneity / energy / correlation on the green channel 4
#   vessel   Frangi vesselness mean, std, and area fraction above Otsu              3
#   lesion   bright (exudate-like) and dark (haemorrhage/microaneurysm-like) top-hat
#            responses: count, area fraction, mean response, each for both polarities 6
#
# DELIBERATELY EXCLUDED: kept-crop area and aspect ratio. Both differ systematically by site
# (DDR arrives square, APTOS retains aspect 1.27-1.31, Messidor-2 keeps ~0.82 of frame), so
# they are site identifiers, not retina features, and a tree model would key on them.
#
# On the lesion proxies. Top-hat counts are a crude lesion proxy, not a detector. DDR ships pixel
# lesion masks for 757 images, so `--validate-lesions` reports the Spearman correlation between
# each proxy and the true per-class lesion area on those images.
#
# Usage:
#     python retina_dr/rdr_data.py features --site all --workers 24
#     python retina_dr/rdr_data.py features --validate-lesions

import rdr_common as P
# (merged: was import rdr_preproc as PRE)

WORK = 512          # features are computed at a fixed size, so scale is not a site cue
GLCM_LEVELS = 32

NAMES = (
    [f'{c}_{s}' for c in 'rgb' for s in ('mean', 'std', 'p10', 'p50', 'p90')]
    + ['glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation']
    + ['vessel_mean', 'vessel_std', 'vessel_frac']
    + ['bright_count', 'bright_area', 'bright_mean',
       'dark_count', 'dark_area', 'dark_mean']
)


def _colour(rgb, fov):
    out = []
    for c in range(3):
        v = rgb[:, :, c][fov].astype(np.float32)
        if v.size == 0:
            out += [0.0] * 5
            continue
        out += [float(v.mean()), float(v.std())] + [float(x) for x in np.percentile(v, [10, 50, 90])]
    return out


def _glcm(green, fov):
    from skimage.feature import graycomatrix, graycoprops
    g = green.copy()
    g[~fov] = 0
    q = (g.astype(np.float32) / 256.0 * GLCM_LEVELS).astype(np.uint8).clip(0, GLCM_LEVELS - 1)
    m = graycomatrix(q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                     levels=GLCM_LEVELS, symmetric=True, normed=True)
    return [float(graycoprops(m, p).mean())
            for p in ('contrast', 'homogeneity', 'energy', 'correlation')]


def _vessel(green, fov):
    from skimage.filters import frangi
    v = frangi(green.astype(np.float32) / 255.0, black_ridges=True)
    vv = v[fov]
    if vv.size == 0:
        return [0.0, 0.0, 0.0]
    thr = float(np.percentile(vv, 90))
    return [float(vv.mean()), float(vv.std()), float((vv > thr).mean())]


def tophat_maps(green, fov):
    """Bright and dark top-hat responses, masked to the fundus. Shared with --validate-lesions."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bright = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, k)
    dark = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, k)
    bright = np.where(fov, bright, 0)
    dark = np.where(fov, dark, 0)
    return bright, dark


def _lesion(green, fov):
    out = []
    for resp in tophat_maps(green, fov):
        vals = resp[fov]
        if vals.size == 0:
            out += [0.0, 0.0, 0.0]
            continue
        thr = max(8.0, float(np.percentile(vals, 99)))
        blobs = (resp >= thr).astype(np.uint8)
        n, _lab, stats, _c = cv2.connectedComponentsWithStats(blobs, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([])
        keep = areas[areas >= 5]
        out += [float(len(keep)), float(keep.sum()) / float(fov.sum() + 1e-9),
                float(vals[vals >= thr].mean()) if (vals >= thr).any() else 0.0]
    return out


def features_from_rgb(rgb):
    """The 28 features of one RGB uint8 image of any size; resized to WORK inside."""
    rgb = cv2.resize(rgb, (WORK, WORK), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fov = g > max(1.0, float(g.max()) * BG_FRAC)
    green = rgb[:, :, 1]
    feats = _colour(rgb, fov) + _glcm(green, fov) + _vessel(green, fov) + _lesion(green, fov)
    return np.nan_to_num(np.array(feats, np.float32))


def _load_rgb(site, image_id):
    return cv2.cvtColor(cv2.imread(cache_path(site, image_id), cv2.IMREAD_COLOR),
                        cv2.COLOR_BGR2RGB)


def extract_one(args):
    site, image_id = args
    try:
        return image_id, features_from_rgb(_load_rgb(site, image_id)), None
    except Exception as exc:                                        # noqa: BLE001
        return image_id, np.zeros(len(NAMES), np.float32), f'{exc}'


def _stable_hash(text):
    import hashlib
    return int.from_bytes(hashlib.md5(text.encode()).digest()[:4], 'little')


def extract_one_aug(args, rgb=None):
    """One augmented copy.

    The transform is `rdr_train.colour_crop_aug`, the one the colour-augmented deep
    candidates train with, applied to the cached image before the resize to WORK. Flips and
    rotations are omitted because every feature here is invariant to them. The rng is seeded
    by (seed, copy, image id), so a copy is reproducible. `rgb` is a test hook."""
    site, image_id, copy, seed = args
    try:
        from rdr_train import colour_crop_aug
        rng = np.random.default_rng([int(seed), int(copy), _stable_hash(image_id)])
        if rgb is None:
            rgb = _load_rgb(site, image_id)
        return image_id, features_from_rgb(colour_crop_aug(rgb, rng)), None
    except Exception as exc:                                        # noqa: BLE001
        return image_id, np.zeros(len(NAMES), np.float32), f'{exc}'


def build_site_features(site, workers):
    # (merged: was import rdr_data as D)
    rows = read_manifest_rows(site)
    if not rows:
        print(f'{site}: no manifest')
        return
    payload = [(site, r['image_id']) for r in rows]
    X = np.zeros((len(rows), len(NAMES)), np.float32)
    errs = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (iid, f, err) in enumerate(ex.map(extract_one, payload, chunksize=16)):
            X[i] = f
            if err:
                errs.append((iid, err))
            if (i + 1) % 2000 == 0:
                print(f'  {site} {i + 1}/{len(rows)}', flush=True)
    out = P.ARTIFACTS / 'features' / f'{site}.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(P.assert_rdr_output(out), X=X,
                        ids=np.array([r['image_id'] for r in rows]),
                        grade=np.array([int(r['grade']) for r in rows]),
                        split=np.array([r['split'] for r in rows]),
                        names=np.array(NAMES))
    print(f'{site:10s} X={X.shape} errors={len(errs)} -> {out}', flush=True)
    for iid, e in errs[:5]:
        print(f'    {iid}: {e}')


def build_site_aug(site, workers, copies, seed, aug='colour'):
    """K augmented copies of every TRAINING image's features -> features/<site>_aug-<aug>.npz.

    Training rows only: evaluation always uses the unaugmented `features/<site>.npz`, as the
    deep candidates are evaluated on unaugmented images."""
    # (merged: was import rdr_data as D)
    rows = [r for r in read_manifest_rows(site) if r['split'] == 'train']
    if not rows:
        print(f'{site}: no training rows')
        return
    payload = [(site, r['image_id'], k, seed) for k in range(copies) for r in rows]
    X = np.zeros((len(payload), len(NAMES)), np.float32)
    errs = []
    # One OpenCV thread per worker: with the default per-process pool, 24 workers on 40 cores
    # drove the load average above 200 (64 % system time) and starved the GPU loaders.
    with ProcessPoolExecutor(max_workers=workers, initializer=cv2.setNumThreads,
                             initargs=(1,)) as ex:
        for i, (iid, f, err) in enumerate(ex.map(extract_one_aug, payload, chunksize=16)):
            X[i] = f
            if err:
                errs.append(f'{iid}: {err}')
            if i % 2000 == 0:
                print(f'  {i}/{len(payload)}', flush=True)
    out = P.ARTIFACTS / 'features' / f'{site}_aug-{aug}.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(P.assert_rdr_output(out), X=X,
                        ids=np.array([q[1] for q in payload]),
                        grade=np.array([int(r['grade']) for _ in range(copies) for r in rows]),
                        split=np.array(['train'] * len(payload)),
                        copy=np.array([q[2] for q in payload]), names=np.array(NAMES),
                        aug=aug, copies=copies, seed=seed)
    print(f'{site} aug={aug} copies={copies} seed={seed}: {X.shape} errors={len(errs)} -> {out}',
          flush=True)
    for e in errs[:10]:
        print('  ' + e)


def validate_lesions():
    """Spearman between each top-hat proxy and true DDR lesion area, over the 757 masked images.

    Bright top-hat should track exudates (EX, SE); dark should track haemorrhages and
    microaneurysms (HE, MA).
    """
    from scipy.stats import spearmanr
    rows = []
    for split, (img_dir, _lab) in ddr_lesion_dirs().items():
        mdir = P.CACHE_DIR / 'ddr_lesion' / split
        if not img_dir.is_dir() or not mdir.is_dir():
            continue
        for name in sorted(os.listdir(img_dir)):
            stem = os.path.splitext(name)[0]
            try:
                rgb = cv2.cvtColor(cv2.imread(str(img_dir / name), cv2.IMREAD_COLOR),
                                   cv2.COLOR_BGR2RGB)
            except Exception:                                       # noqa: BLE001
                continue
            if rgb is None:
                continue
            y0, y1, x0, x1 = fundus_bbox(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            crop = resize_long(rgb[y0:y1, x0:x1])
            crop = cv2.resize(crop, (WORK, WORK), interpolation=cv2.INTER_AREA)
            g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            fov = g > max(1.0, float(g.max()) * BG_FRAC)
            bright, dark = tophat_maps(crop[:, :, 1], fov)
            rec = {'stem': stem}
            for pol, resp in (('bright', bright), ('dark', dark)):
                vals = resp[fov]
                thr = max(8.0, float(np.percentile(vals, 99)))
                rec[f'proxy_{pol}'] = float((resp >= thr).sum()) / float(fov.sum() + 1e-9)
            ok = True
            for cls in LESION_CLASSES:
                mp = mdir / cls / f'{stem}.png'
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp.is_file() else None
                if m is None:
                    ok = False
                    break
                m = cv2.resize(m, (WORK, WORK), interpolation=cv2.INTER_NEAREST)
                rec[f'true_{cls}'] = float((m > 0).sum()) / float(m.size)
            if ok:
                rows.append(rec)
    print(f'validated on {len(rows)} masked DDR images')
    if len(rows) < 30:
        print('too few masked images to report a correlation')
        return
    pairs = [('proxy_bright', ['EX', 'SE']), ('proxy_dark', ['HE', 'MA'])]
    for proxy, classes in pairs:
        pv = np.array([r[proxy] for r in rows])
        for cls in classes:
            tv = np.array([r[f'true_{cls}'] for r in rows])
            rho, p = spearmanr(pv, tv)
            print(f'  {proxy:13s} vs true {cls:2s} area   rho={rho:+.3f}  p={p:.2e}')
        tot = np.sum([[r[f'true_{c}'] for c in classes] for r in rows], axis=1)
        rho, p = spearmanr(pv, tot)
        print(f'  {proxy:13s} vs {"+".join(classes):7s} total  rho={rho:+.3f}  p={p:.2e}')


def main_features(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', default='all', choices=('all', 'ddr', 'aptos', 'messidor2'))
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--validate-lesions', action='store_true')
    ap.add_argument('--aug', choices=('colour',), default=None,
                    help='write K augmented copies of the TRAINING features instead '
                         '(colour jitter + random crop, as rdr_train.py deep --aug colour)')
    ap.add_argument('--copies', type=int, default=4, help='K for --aug')
    ap.add_argument('--seed', type=int, default=0, help='rng seed for --aug')
    args = ap.parse_args(argv)
    if args.validate_lesions:
        validate_lesions()
        return
    sites = ('ddr', 'aptos', 'messidor2') if args.site == 'all' else (args.site,)
    for s in sites:
        if args.aug:
            build_site_aug(s, args.workers, args.copies, args.seed, args.aug)
        else:
            build_site_features(s, args.workers)


# ==========================================================================================
# rdr_views.py
# ==========================================================================================
# rdr_views.py -- the controlled input views (the view experiment).
#
# Each view is a deterministic function of the cached fundus image, applied
# before augmentation and resizing, identically at every site, with no site or label input:
#
#   rgb     unchanged (the `retina-whole-view` input)
#   green   green channel, CLAHE (clip 2.0, 8 x 8 grid), copied to three channels
#   vessel  Frangi vesselness of the green channel at 512 px, as `rdr_features._vessel`,
#           scaled by its 99th percentile inside the field of view, three channels
#   crop    central square of side half the shorter image side
#
# Frangi costs about 1 s per image, so the vessel view is precomputed once into
# `cache_vessel/<site>/<id>.png` and read back at load time.
#
# Usage:
#     python retina_dr/rdr_data.py views --build-vessel --site all --workers 30

import rdr_common as P
# (merged: was import rdr_preproc as PRE)

VIEWS = ('rgb', 'green', 'vessel', 'crop')
VESSEL_DIR = P.ARTIFACTS / 'cache_vessel'
VESSEL_SIZE = 512           # the WORK size of the features section
CLAHE_CLIP, CLAHE_GRID = 2.0, (8, 8)
CROP_FRAC = 0.5


def vessel_path(site, image_id):
    return str(VESSEL_DIR / site / (os.path.splitext(image_id)[0] + '.png'))


def vessel_map(rgb):
    """[H, W, 3] uint8 RGB -> [512, 512, 3] uint8 vesselness view."""
    from skimage.filters import frangi
    rgb = cv2.resize(rgb, (VESSEL_SIZE, VESSEL_SIZE), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fov = g > max(1.0, float(g.max()) * BG_FRAC)
    v = frangi(rgb[:, :, 1].astype(np.float32) / 255.0, black_ridges=True)
    v = np.where(fov, v, 0.0)
    scale = float(np.percentile(v[fov], 99)) if fov.any() else 0.0
    v = np.clip(v / scale, 0.0, 1.0) if scale > 0 else np.zeros_like(v)
    return np.repeat((v * 255.0 + 0.5).astype(np.uint8)[:, :, None], 3, axis=2)


def apply(view, rgb, site, image_id):
    """The view of one cached image. `rgb` is returned unchanged for 'rgb'."""
    if view == 'rgb':
        return rgb
    if view == 'green':
        g = cv2.createCLAHE(CLAHE_CLIP, CLAHE_GRID).apply(np.ascontiguousarray(rgb[:, :, 1]))
        return np.repeat(g[:, :, None], 3, axis=2)
    if view == 'vessel':
        bgr = cv2.imread(vessel_path(site, image_id), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f'vessel view missing for {site}/{image_id}: run --build-vessel')
        return bgr[:, :, ::-1].copy()
    if view == 'crop':
        h, w = rgb.shape[:2]
        side = int(round(min(h, w) * CROP_FRAC))
        y0, x0 = (h - side) // 2, (w - side) // 2
        return np.ascontiguousarray(rgb[y0:y0 + side, x0:x0 + side])
    raise ValueError(f'unknown view {view}')


def _one_vessel(args):
    site, image_id = args
    dst = vessel_path(site, image_id)
    if os.path.exists(dst):
        return 'cached'
    bgr = cv2.imread(cache_path(site, image_id), cv2.IMREAD_COLOR)
    if bgr is None:
        return f'unreadable {site}/{image_id}'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    out = vessel_map(bgr[:, :, ::-1])
    return 'ok' if cv2.imwrite(dst, out[:, :, ::-1]) else f'write failed {dst}'


def build_vessel(sites, workers):
    # (merged: was import rdr_data as D)
    P.assert_rdr_output(VESSEL_DIR)
    for site in sites:
        jobs = [(site, r['image_id']) for r in read_manifest_rows(site)]
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_one_vessel, jobs, chunksize=16))
        errs = [r for r in res if r not in ('ok', 'cached')]
        print(f'{site}: n={len(res)} ok={res.count("ok")} cached={res.count("cached")} '
              f'errors={len(errs)}' + (f' first={errs[0]}' if errs else ''), flush=True)
        if errs:
            raise SystemExit(f'vessel cache incomplete for {site}')


def main_views(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-vessel', action='store_true')
    ap.add_argument('--site', default='all')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args(argv)
    if args.build_vessel:
        sites = ('ddr', 'aptos', 'messidor2', 'deepdrid') if args.site == 'all' else (args.site,)
        build_vessel(sites, args.workers)


COMMANDS = {'manifests': main_manifests, 'preproc': main_preproc, 'features': main_features, 'views': main_views}


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: {os.path.basename(sys.argv[0])} {'|'.join(COMMANDS)} [options]")
    COMMANDS[sys.argv[1]](sys.argv[2:])
