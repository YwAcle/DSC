# Dynamic-Static Collaboration (DSC) for UDA Video RGB–IR Re-ID

Code for the paper **“Dynamic-Static Collaboration for Unsupervised Domain Adaptive Video-Based Visible-Infrared Person Re-Identification.”**

> ⏳ Code will be released after cleanup. Please star/watch this repo.

## 1. Overview
- Task: unsupervised domain-adaptive, **video-based**, **RGB–IR** person re-ID.
- Idea: extract **static** (appearance) and **dynamic** (temporal) features separately, then make them **agree** on pseudo labels.
- Modules:
  - **DSLU**: dynamic–static label unification.
  - **DSJL**: dynamic–static joint learning.

## 2. Datasets
- **HITSZ-VCM**
- **BUPT-Campus**
(we will provide folder structure + sample configs)

## 3. Training (placeholder)
```bash
python tools/train_source.py --cfg configs/source_vcm.yaml
python tools/generate_pseudo_labels.py --cfg configs/uda_vvi_reid.yaml --with-dslu
python tools/train_uda_dsc.py --cfg configs/uda_vvi_reid.yaml --with-dslu --with-dsjl
python tools/test.py --cfg configs/uda_vvi_reid.yaml --evaluate
