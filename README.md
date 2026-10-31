# Explainable AI for Pneumonia Detection from Chest X-Rays

## Description
This project investigates the reliability of two explainable AI (XAI) techniques,
Grad-CAM and LIME, for pneumonia detection from chest X-ray images. The study is
conducted in two stages: (1) binary classification (Normal vs. Pneumonia) using
ResNet-50, and (2) three-class classification (Normal vs. Bacterial vs. Viral
pneumonia) using DenseNet-121, with a focus on applicability in
resource-constrained healthcare settings such as Sri Lanka.

## Team
- S. Niroshini (ITBIN-2313-0074): binary pipeline (ResNet-50), Grad-CAM/LIME comparison
- T. Deva Nivethitha (ITBIN-2313-0075): three-class pipeline (DenseNet-121)

**Module:** IT41043 Intelligent Systems | Horizon Campus | Third Year, Second Semester 2026

## Datasets
- **Kermany Chest X-Ray Dataset** (public benchmark, 5,856 images; used so far)
- **Locally collected Sri Lankan chest X-ray dataset** (planned; not collected yet,
  subject to ethics approval)

## Research Question (as proposed)

> Does a three-class DenseNet-121 model (Normal vs. Bacterial vs. Viral Pneumonia)
> integrated with Grad-CAM produce visual explanations that enable clinicians to
> distinguish bacterial from viral cases with statistically significantly higher
> accuracy compared to using model predictions without explanations, and does
> Grad-CAM outperform LIME on this task?

**Falsifiability:**
1. If Grad-CAM explanations do not significantly improve clinicians' accuracy in
   distinguishing bacterial vs. viral pneumonia compared to no explanations, the
   claim is falsified.
2. If Grad-CAM does not significantly outperform LIME in clinician accuracy or
   confidence, the claim is falsified.

**Note:** Clinician-based evaluation has not been carried out yet. The results below
use quantitative proxy metrics only (see Limitations).

## Project Structure
```
data/            - raw and processed datasets (not tracked in git)
preprocessing/   - early standalone preprocessing scripts
notebooks/       - executable pipelines (training, Grad-CAM, LIME, evaluation)
xai/             - XAI code
results/         - figures and evaluation outputs
docs/            - architecture diagrams
```

## Installation
```bash
git clone https://github.com/Niroshini2002/pneumonia-xai-gradcam-lime.git
cd pneumonia-xai-gradcam-lime
pip install -r requirements.txt
```

## Usage
The executable pipeline is the notebook `notebooks/ResNet50_Binary_Training.ipynb`
(run on Google Colab, T4 GPU). It downloads the Kermany dataset through the Kaggle
API (credentials stored as Colab Secrets, never in the repo), preprocesses the
images, trains ResNet-50, and runs Grad-CAM and LIME.

`preprocessing/preprocess_binary.py` is an earlier standalone script and is not the
pipeline used for the reported results. Three-class training code is in development.

Model weights (`resnet50_binary.pth) are not stored in this repository:
https://drive.google.com/file/d/1M5eQhDXLSvlqt1KR3oPMGpZEVqj9a0qs/view?usp=sharing

## Current Status
In progress. Binary ResNet-50 pipeline, 5-fold cross-validation and the Grad-CAM vs
LIME proxy comparison are complete. The three-class DenseNet-121 pipeline is in
development.

## Results: Binary Classification (ResNet-50, Kermany dataset)

Patient-level stratified split (80/10/10); patient IDs are inferred from filenames and
no patient appears in more than one split.

| Evaluation | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Held-out test set (n = 615) | 0.9837 | 0.9892 | 0.9903 |
| 5-fold stratified group CV (mean ± SD) | 0.9681 ± 0.0067 | 0.9778 ± 0.0049 | 0.9958 ± 0.0014 |

Test confusion matrix: [[149, 8], [2, 456]]. The test set is 74.5% pneumonia, so
accuracy should be read together with specificity (94.9%). CV SDs are sample SDs
(ddof = 1) computed from fold values rounded to four decimals.

## Results: Grad-CAM vs LIME (proxy metrics)

n = 100 correctly predicted pneumonia test images, top 20% of each heatmap.

| Metric | Grad-CAM | LIME | Random |
|---|---|---|---|
| Deletion drop (mean) | 0.030 | 0.084 | 0.032 |
| Deletion drop (median) | 0.001 | 0.011 | 0.005 |
| Edge energy (mean) | 0.424 | 0.544 | n/a |

Inter-method IoU: 0.246 (chance level about 0.118). On these proxies, Grad-CAM was not
better than LIME and was not distinguishable from random deletion; LIME was higher
than random but not significantly so after Bonferroni correction (p = 0.055).

## Limitations
- Single dataset (Kermany, one pediatric centre); no local Sri Lankan data yet
- Single model, single seed; proxy faithfulness metrics only
- Clinician agreement and localization accuracy against radiologist annotations
  were not measured
- Deletion metric uses black-pixel fill, which may be out-of-distribution
- Patient IDs inferred from filenames

## Evaluation Metrics
- Accuracy, Precision, Recall, F1-score, AUC-ROC (classification)
- Deletion drop, edge energy, inter-method IoU (explanation proxies)
- Stratified group k-fold cross-validation
- Wilcoxon signed-rank test with Bonferroni correction
