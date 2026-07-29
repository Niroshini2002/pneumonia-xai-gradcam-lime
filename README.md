# Explainable AI for Pneumonia Detection from Chest X-Rays

## Description
This project investigates the reliability of two explainable AI (XAI) techniques — 
Grad-CAM and LIME — for pneumonia detection from chest X-ray images. The study is 
conducted in two stages: (1) binary classification (Normal vs. Pneumonia) using 
ResNet-50, and (2) three-class classification (Normal vs. Bacterial vs. Viral 
pneumonia) using DenseNet-121. Explanation quality is evaluated using clinician 
agreement, localization accuracy, and diagnostic confidence metrics, with a focus 
on applicability in resource-constrained healthcare settings such as Sri Lanka.

## Team
- S. Niroshini (ITBIN-2313-0074)
- T. Deva Nivethitha (ITBIN-2313-0075)

**Module:** IT41043 — Intelligent Systems | Horizon Campus | Third Year, Second Semester 2026

## Datasets
- **Kermany Chest X-Ray Dataset** (public benchmark)
- **Locally collected Sri Lankan chest X-ray dataset**

## Project Structure
data/ - raw and processed datasets
preprocessing/ - data cleaning and preparation scripts
models/ - binary and three-class classification models
xai/ - Grad-CAM and LIME implementation
evaluation/ - metrics and statistical significance tests
notebooks/ - exploratory analysis and results
results/ - output visualizations and evaluation reports
docs/ - architecture diagrams

## Installation
```bash
git clone <repo-url>
cd pneumonia-xai-gradcam-lime
pip install -r requirements.txt
```

## Usage
```bash
# Preprocess data
python preprocessing/preprocess_binary.py
python preprocessing/preprocess_threeclass.py

# Run Grad-CAM / LIME explanations
python xai/gradcam.py
python xai/lime_explain.py
```

## Current Status
🔧 In Progress — Milestone 2: Preprocessing scripts and initial folder structure 
completed. Model training and XAI evaluation pipeline in development.

## Evaluation Metrics
- Accuracy, F1-score, AUC-ROC (classification)
- Clinician agreement score, localization accuracy (explanation quality)
- Stratified k-fold cross-validation
- Statistical significance testing (paired t-test / Wilcoxon signed-rank test)

## License
Academic project — Horizon Campus, IT41043.
