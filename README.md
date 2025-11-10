# Dynamic-Static Collaboration (DSC) for Unsupervised Domain-Adaptive Video-Based Visible-Infrared Re-ID

This repository will release the official implementation of our paper  
**“Dynamic-Static Collaboration for Unsupervised Domain Adaptive Video-Based Visible-Infrared Person Re-Identification (DSC)”**.

The paper introduces a new task, **unsupervised domain-adaptive video-based visible-infrared person re-ID (UDA VVI-ReID)**, and proposes a **Dynamic-Static Collaboration** framework to tackle unreliable pseudo labels caused by cross-modality gaps and temporal noise. 📹🌗

> ⏳ **Status:** Code is being organized and will be released after the paper is publicly available. Please ⭐️ or 👀 this repo to get notified.

---

## 📄 Paper

- **Title:** Dynamic-Static Collaboration for Unsupervised Domain Adaptive Video-Based Visible-Infrared Person Re-Identification  
- **Problem setting:** Source domain is labeled, target domain is unlabeled, both are video-based and cross-modality (RGB–IR).  
- **Key idea:** Separately model **static (appearance)** and **dynamic (temporal/motion)** cues, let them **cross-check** each other to filter/rectify pseudo labels, and then jointly learn more robust cross-modality representations.
- **Main components:**
  - **DSLU (Dynamic-Static Label Unification):** Unifies and corrects pseudo labels by enforcing consistency between static and dynamic branches.
  - **DSJL (Dynamic-Static Joint Learning):** Performs neighbor-aware contrastive / metric learning in dual feature spaces to improve robustness.

(You can find the paper PDF in this repo, e.g. `./DSC.pdf`.)

---

## 🧠 Method Overview

Typical video Re-ID features are obtained by simple temporal pooling, which often mixes useful static cues (color/shape/structure) and dynamic cues (motion/temporal consistency) and weakens both.

DSC does it in two stages:

1. **Decouple first:** extract static and dynamic features separately, and perform similarity/pseudo-label generation in both spaces;
2. **Cross-validate later:** only keep or refine those pseudo labels that are consistent in both branches, so that unlabeled target-domain training becomes more reliable.

This makes unsupervised domain adaptation on video RGB–IR more stable and leads to better results on benchmarks such as **HITSZ-VCM** and **BUPT-Campus** (see paper for details).

---

## 📦 Datasets

We evaluate on two video-level visible–infrared datasets:

- **HITSZ-VCM**
  - Video tracklets with both RGB and IR
  - Around a few dozen frames per tracklet
  - Used for video-level cross-modality Re-ID

- **BUPT-Campus**
  - Larger-scale RGB–IR video dataset with more IDs and more tracklets
  - Better for testing UDA generalization

When the code is released, we will provide:

- expected folder structure
- scripts to convert/organize the datasets
- example config files for each dataset

---

## 🛠 Environment (to be provided)

The released code will be based on a standard deep learning stack, for example:

- Python ≥ 3.8
- PyTorch ≥ 1.12 / 2.x
- CUDA 11.x
- Common Re-ID / SSL utilities (tqdm, yaml/yacs/jsonargparse, FAISS or sklearn for clustering)

A `requirements.txt` or `environment.yml` will be included.

---

## 🚀 Training & Evaluation (placeholder)

> ⚠️ The following commands are **examples**. Actual script names, arguments, and config paths will be finalized in the release.

```bash
# 1. Supervised training on the labeled source domain (RGB + IR)
python tools/train_source.py \
  --cfg configs/source_vcm.yaml

# 2. Generate dynamic-static-consistent pseudo labels on the unlabeled target domain (DSLU)
python tools/generate_pseudo_labels.py \
  --cfg configs/uda_vvi_reid.yaml \
  --with-dslu

# 3. Joint learning with both dynamic and static branches (DSJL)
python tools/train_uda_dsc.py \
  --cfg configs/uda_vvi_reid.yaml \
  --with-dslu --with-dsjl

# 4. Evaluate on the target dataset
python tools/test.py \
  --cfg configs/uda_vvi_reid.yaml \
  --evaluate
