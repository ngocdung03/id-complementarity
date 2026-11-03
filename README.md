# Choosing the Combination Operator and the Added Model from In-Distribution Data for Fundus Grading under Acquisition Shift

Code for the ISBI 2027 submission.

Ngoc-Dung Nguyen, Jong-Hoon Lee, Ha-Hieu Pham, Thanh-Huy Nguyen, Xiaofeng Liu, Ulas Bagci,
Hyunsong Jang.

A fixed anchor model for diabetic retinopathy grading is combined with one added candidate
model at low weight (w = 0.15), by arithmetic or geometric averaging of the grade
probabilities. The code trains the anchor and the candidates on DDR, scores them on four
out-of-distribution (OOD) fundus collections, computes each candidate's net correction of the
anchor on in-distribution (ID) data, and runs the controlled view experiment.

## Pipeline and selection sets

| Step | Data used | What is chosen |
|---|---|---|
| Training of the anchor and of every candidate | DDR train | model weights |
| Training recipe (learning rate, trunk-to-head rate ratio, weight decay) | DDR valid, best of ten trials per model | one recipe per model |
| Combination weight | none | fixed, w = 0.15 for every candidate and operator |
| Everything reported | DDR test (ID), APTOS 2019, Messidor-2, DeepDRiD, IDRiD (OOD) | nothing; evaluation only |

## Layout

| File | Contents |
|---|---|
| `retina_dr/rdr_common.py` | paths (`RDR_ROOT`), run tags, cut scores, geometric and arithmetic combination, strict alignment, pair statistics, net correction; `operator` (combination table at w = 0.15, 0.20, 0.25) |
| `retina_dr/rdr_data.py` | `manifests`, `preproc` (field-of-view crops, lesion masks), `features` (28 handcrafted features), `views` (the controlled input views) |
| `retina_dr/rdr_models.py` | the patch-bag anchor and the whole-image candidates (timm backbones, cumulative ordinal head) |
| `retina_dr/rdr_train.py` | `deep` (anchor and deep candidates), `feat` (GBM, RF, SVM), `lesion` (lesion detector used for the site features), `tune` (recipe search on DDR valid) |
| `retina_dr/rdr_predict.py` | `site` (DeepDRiD and IDRiD manifests, features, predictions), `idrid-views`, `boot` (DDR bootstrap draws for the view runs), `viewstats` (view-run statistics) |
| `retina_dr/rdr_report.py` | Tables 1 and 2 and the numbers of Sections 3.1 to 3.3 |
| `figures/make_fig_views.py` | the controlled-view figure |

## Setup

Python 3.11. The pinned versions in `requirements.txt` are those of the environment the
results were produced with (torch 2.6, CUDA 12.4; CPU works for the analyses).

```bash
pip install -r requirements.txt
export RDR_ROOT=/path/to/workdir     # default: this repository; data under $RDR_ROOT/data
```

## Data

Five public colour-fundus collections with ICDR grades 0-4: DDR (training and ID test),
APTOS 2019, Messidor-2 (adjudicated grades of Krause et al., 2018), DeepDRiD (regular fundus
images) and IDRiD (disease grading). Download them from their providers into `$RDR_ROOT/data/`
(the site roots are listed in `retina_dr/rdr_common.py`). The datasets keep their own
licences; obtain them from the original providers. DDR is split by its official directories
after removing ungradable images and cross-split duplicates (`rdr_data.py manifests`).

## Reproducing the paper

Run from the repository root. Multi-step scripts take a subcommand first; every
result-producing step takes `--kw <keyword>`, which becomes the run tag `retina-<keyword>`.

```bash
# 1. Manifests, preprocessing, handcrafted features, lesion detector
python retina_dr/rdr_data.py manifests
python retina_dr/rdr_data.py preproc --site all --workers 16
python retina_dr/rdr_data.py features --site all --workers 24
python retina_dr/rdr_train.py lesion --kw les2

# 2. Recipe search on DDR valid: ten trials per model, trial 0 = the original recipe
for m in patchbag whole convnext effnet dinov2; do
  python retina_dr/rdr_train.py tune --member $m --kw tune --trials 10 --epochs 12
done
# Best trials: patchbag t00 (anchor), whole t00 (ResNet-50), convnext t08, effnet t03, dinov2 t02.
python retina_dr/rdr_train.py feat --clf all --kw feat        # GBM / RF / SVM, fixed settings

# 3. Predictions of the candidates on DeepDRiD and IDRiD
M=retina-convnext-tune-t08,retina-whole-view,retina-effnet-tune-t03,retina-dinov2-tune-t02,retina-svm-feat,retina-rf-feat,retina-gbm-feat
for site in deepdrid idrid; do
  python retina_dr/rdr_predict.py site prepare --site $site --kw tier1
  python retina_dr/rdr_predict.py site predict --site $site --kw tier1 --members $M      # anchor included
done

# 4. Controlled views: 4 views x 3 seeds at the ResNet-50 recipe
python retina_dr/rdr_data.py views --build-vessel --site all --workers 30
for v in rgb green vessel crop; do for s in 0 1 2; do
  python retina_dr/rdr_train.py deep --member whole --view $v --seed $s --epochs 12 --trunk-div 1 \
      --extra-site deepdrid --blind --kw $v-t2s$s
done; done
python retina_dr/rdr_predict.py idrid-views --kw idrid-views
python retina_dr/rdr_predict.py boot --kw boot
python retina_dr/rdr_predict.py viewstats --kw tier2

# The same twelve runs with colour jitter and random-resized crop added to the augmentation
for v in rgb green vessel crop; do for s in 0 1 2; do
  python retina_dr/rdr_train.py deep --member whole --view $v --seed $s --epochs 12 --trunk-div 1 \
      --extra-site deepdrid --blind --aug colour --kw $v-t2cs$s
done; done
python retina_dr/rdr_predict.py viewstats --kw tier2-colour --suffix t2cs

# 5. Tables 1 and 2, Sections 3.1 to 3.3
python retina_dr/rdr_report.py --kw tuned --members $M --boot retina-boot

# 6. Weight sweep (Sec. 3.1, w = 0.20 and 0.25)
python retina_dr/rdr_common.py operator --anchor retina-patchbag-anchor --members $M --kw all7

# Figure
python figures/make_fig_views.py
```

Outputs are written under `$RDR_ROOT/artifacts/` and are never overwritten; a re-run needs a new `--kw`.

## Licence

Code: MIT (see `LICENSE`). The datasets keep their own licences.
