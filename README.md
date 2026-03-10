# DeepMSM


*Scientific Reports (2026) 16:4037* · [Paper](https://doi.org/10.1038/s41598-025-34134-9)

![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)

---

## Overview

Survival prediction in IDH-wildtype glioblastoma is challenged by **inter-institutional heterogeneity** and **scarce labeled data**. FA-DeepMSM addresses these challenges by combining:

- **Self-supervised MRI embeddings** via a pretrained DINOv2 Vision Transformer (ViT)
- **Structured clinical & molecular variables** (age, KPS, EOR, IDH, MGMT, WHO grade)
- **Few-shot fine-tuning** (0 / 5 / 10 / 20 / 40 shots) for cross-institutional adaptation
- **Time-resolved interpretability** via permutation-based feature importance at 3, 6, 12, 18, and 24 months

| Metric | Value |
|---|---|
| Internal C-index (multimodal DeepMSM) | **0.788** (95% CI: 0.782–0.793) |
| External C-index — 0-shot | 0.643 (95% CI: 0.638–0.647) |
| External C-index — 40-shot (FA-DeepMSM) | **0.680** (95% CI: 0.676–0.683) |
| Relative improvement (0→40 shot) | **+5.7%** |
| Training cohort size | 1,359 adult-type diffuse glioma patients |
| External validation cohort | UPenn-GBM (n = 452) |

---

## Why It Matters

Existing multimodal survival models frequently **fail to generalize** when applied to external institutions due to:

1. **Domain shift** in MRI acquisition protocols and scanner variability
2. **Overfitting** of high-dimensional imaging features to small labeled datasets
3. **Limited interpretability** of black-box fusion architectures

FA-DeepMSM tackles all three simultaneously: a frozen self-supervised encoder minimizes overfitting, few-shot linear probing enables rapid adaptation, and a time-resolved permutation module reveals *when* each variable matters most across the survival timeline.

---

## Model Architecture


![image](https://github.com/user-attachments/assets/f08fd2de-e9c6-4942-ac04-a26390fddc8b)
```
┌──────────────────────────────────────────────────────────────┐
│                        PRE-TRAIN PHASE                        │
│                                                               │
│   3D Multi-parametric MRI  ──►  DINOv2 ViT (self-supervised) │
│   (T1, T1CE, T2, FLAIR)         Teacher-Student + EMA        │
│                                 Cross-Entropy + KoLeo Loss    │
└──────────────────────┬───────────────────────────────────────┘
                       │  Frozen MRI Encoder (DPIs)
┌──────────────────────▼───────────────────────────────────────┐
│                     DOWNSTREAM PHASE                          │
│                                                               │
│  DL-MRI Features (f_I) ──┐                                   │
│                           ├──► Concat ──► Multi-modal Fusion  │
│  Clinical Features (f_T) ─┘              Transformer          │
│  (Age, KPS, EOR, IDH,                        │               │
│   MGMT, WHO grade, sex)                      ▼               │
│                                    Hazard Estimator (MLP)     │
│                                              │               │
│                                              ▼               │
│                             Survival Prediction: λ₁ … λ_Tmax │
└──────────────────────────────────────────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │     FEW-SHOT ADAPTATION   │
         │  FA-DeepMSM (n=5/10/20/40)│
         │  Linear probing of final  │
         │  projection layer only    │
         └───────────────────────────┘
```

### Key Components

**Image Encoder** — DINOv2-pretrained 3D Vision Transformer. Sequence-wise averaged embeddings (T1, T1CE, T2, FLAIR) form the Deep Prognostic Index (DPI). The encoder is frozen during few-shot adaptation.

**Modality Fusion Layer** — Concatenates DPI embeddings with structured tabular clinical variables. A lightweight transformer-based fusion avoids the proportional hazards constraint of CoxPH while keeping parameter count low.

**Hazard Network** — Shared MLP with sigmoid activation outputs time-dependent conditional hazard estimates across predefined time horizons. Trained with negative log-likelihood loss accounting for right-censored patients.

**Few-Shot Adaptation** — Only the final projection layer is fine-tuned on n = {5, 10, 20, 40} samples from the external target domain. All other layers remain frozen to prevent overfitting.

**Time-Resolved Interpretability** — Permutation-based feature importance quantified as ΔC-index at each evaluation time point (3, 6, 12, 18, 24 months) for all input variables and MRI-derived DPIs.

---

## Performance

### Internal Validation

| Model | Data | Avg. C-index |
|---|---|---|
| CoxPH | Clinical | 0.749 |
| CoxPH | Multimodal | 0.749 |
| RSF | Multimodal | 0.727 |
| **DeepMSM** | **Multimodal** | **0.788** |

### External Validation (UPenn-GBM, 12-month C-index)

| Model | 0-shot | 5-shot | 10-shot | 20-shot | 40-shot |
|---|---|---|---|---|---|
| CoxPH (Multimodal) | 0.631 | 0.616 | 0.618 | 0.642 | 0.641 |
| RSF (Multimodal) | 0.611 | 0.456 | 0.539 | 0.582 | 0.607 |
| **FA-DeepMSM (Multimodal)** | **0.642** | 0.575 | 0.653 | 0.669 | **0.674** |

All pairwise comparisons against Multimodal FA-DeepMSM: p < 0.001.

---

## Key Findings

### Time-Dependent Feature Importance

| Feature | Peak Importance | Clinical Interpretation |
|---|---|---|
| **Extent of Resection (EOR)** | Early (3–12 months) | Drives short-term post-surgical outcomes |
| **MGMT Methylation** | Late (≥18 months) | Reflects chemotherapy sensitivity |
| **Age** | Consistently higher in GBM | Established independent GBM prognostic factor |
| **MRI-DPI** | Increases with shot count | Imaging aligns progressively via few-shot adaptation |

### Few-Shot Adaptation Dynamics

- At **0–5 shots**: imaging features contribute negatively or neutrally; clinical variables dominate
- From **10 shots onward**: DPI ΔC-index rises consistently, demonstrating effective image–clinical alignment
- CoxPH shows **no recovery** of imaging importance across all shot levels, highlighting the advantage of deep learning

### Patient-Level Interpretability (Grad-CAM)

Grad-CAM activations align with manually annotated tumor regions (enhancing tumor + peritumoral edema) without using segmentation masks as input — supporting the biological plausibility of learned representations.

---

## Dataset

| Cohort | Institution | n | Role |
|---|---|---|---|
| SNUH | Seoul National University Hospital | 802 | Internal training |
| Severance | Severance Hospital | 87 | Internal training |
| UCSF-PDGM | Univ. of California, San Francisco | 470 | Internal training |
| **UPenn-GBM** | **Univ. of Pennsylvania** | **452** | **External validation** |

- **Internal cohorts** include IDH-wildtype GBM, IDH-mutant astrocytoma, and IDH-mutant oligodendroglioma (2021 WHO classification)
- **External cohort** is composed exclusively of IDH-wildtype glioblastoma
- MRI sequences: T1, T1CE, T2, FLAIR (skull-stripped, 1 mm isotropic, co-registered, intensity-normalized)
- Clinical variables: age, sex, KPS, EOR, WHO grade, IDH status, MGMT status, glioma pathology type

> **Note:** The in-house SNUH and Severance datasets are not publicly available. Restricted access may be requested from the corresponding author. UCSF-PDGM and UPenn-GBM are publicly available datasets.

---

## Installation

```bash
git clone https://github.com/ggomaeng514/DeepMSM.git
cd DeepMSM
pip install -r requirements.txt
```
---

## Citation

If you find FA-DeepMSM useful, please cite:

```bibtex
@article{hwang2026fadeepmsm,
  title     = {FA-DeepMSM: a few-shot adapted interpretable multimodal survival model
               for improved prognostic prediction in glioblastoma},
  author    = {Hwang, Minyoung and Lee, Junhyeok and Kim, Sihyeon and Kim, Minchul
               and Choi, Seung Hong and Ahn, Sung Soo and Lee, Changhee and Choi, Kyu Sung},
  journal   = {Scientific Reports},
  volume    = {16},
  pages     = {4037},
  year      = {2026},
  publisher = {Nature Publishing Group},
  doi       = {10.1038/s41598-025-34134-9}
}
```

---

## Authors

| Name | Affiliation | Role |
|---|---|---|
| Minyoung Hwang | Korea University, Dept. of AI | First author, study design, model development |
| Junhyeok Lee | Seoul National University, Cancer Biology | Co-first author, study design, statistical analysis |
| Sihyeon Kim | Korea University, Dept. of AI | Model development |
| Minchul Kim | Kangbuk Samsung Hospital, Radiology | Clinical data curation |
| Seung Hong Choi | SNUH, Radiology | Clinical supervision |
| Sung Soo Ahn | Yonsei University, Radiology | Clinical data curation |
| **Changhee Lee** ✉ | Korea University, Dept. of AI | Co-corresponding, technical oversight |
| **Kyu Sung Choi** ✉ | SNUH, Radiology | Co-corresponding, clinical supervision |

✉ Corresponding authors: [changheelee@korea.ac.kr](mailto:changheelee@korea.ac.kr) · [ent1127@snu.ac.kr](mailto:ent1127@snu.ac.kr)

---

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).

You may share this work for non-commercial purposes with attribution, but may not distribute adapted or modified versions.

---

## Ethics

Data collection was approved by the Institutional Review Boards of Seoul National University Hospital (SNUH) and Severance Hospital, with informed consent waived. All procedures followed the Declaration of Helsinki. Analyses of de-identified public datasets (UCSF-PDGM, UPenn-GBM) were exempt from IRB approval under institutional policies for secondary use of anonymized data.
