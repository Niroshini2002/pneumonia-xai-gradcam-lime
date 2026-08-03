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

## 🔬 Research Question

> *Does a three-class DenseNet-121 model (Normal vs. Bacterial Pneumonia vs. Viral Pneumonia) integrated with Grad-CAM produce visual explanations that enable clinicians to distinguish bacterial from viral cases with statistically significantly higher accuracy compared to using model predictions without explanations?*

**Falsifiability:**
1. If Grad-CAM explanations do not significantly improve clinicians' accuracy in distinguishing bacterial vs. viral pneumonia compared to no explanations, the claim is falsified.
2. If Grad-CAM does not significantly outperform LIME in clinician accuracy or confidence, the claim is falsified.

---

## Project Structure
data/ - raw and processed datasets
preprocessing/ - data cleaning and preparation scripts
models/ - binary and three-class classification models
xai/ - Grad-CAM and LIME implementation
evaluation/ - metrics and statistical significance tests
notebooks/ - exploratory analysis and results
results/ - output visualizations and evaluation reports
docs/ - architecture diagrams

pneumonia-xai-gradcam-lime/
│
├── data/
│ ├── raw/ # Raw dataset storage
│ │ └── kermany/ # Kermany dataset
│ └── processed/ # Preprocessed data splits
│
├── preprocessing/
│ ├── preprocess_binary.py # Binary classification preprocessing
│ └── preprocess_threeclass.py # Three-class preprocessing
│
├── models/
│ ├── resnet50_binary.py # ResNet-50 for binary classification
│ ├── densenet121_threeclass.py # DenseNet-121 for three-class
│ └── train.py # Training script
│
├── xai/
│ ├── gradcam.py # Grad-CAM implementation
│ └── lime_explain.py # LIME implementation
│
├── evaluation/
│ ├── metrics.py # Evaluation metrics
│ └── statistical_tests.py # Statistical significance tests
│
├── notebooks/
│ ├── 01_eda.ipynb # Exploratory data analysis
│ ├── 02_preprocessing.ipynb # Preprocessing demonstration
│ └── 03_visualization.ipynb # Results visualization
│
├── docs/
│ └── architecture_diagram.svg # System architecture diagram
│
├── results/
│ ├── figures/ # Generated figures and plots
│ └── reports/ # Evaluation reports
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE

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

## Model Training

# Train ResNet-50 for binary classification
python models/train.py --model resnet50 --task binary

# Train DenseNet-121 for three-class classification
python models/train.py --model densenet121 --task threeclass## Model Architecture

## Model Architecture
# System Architecture Diagram

 ┌─────────────────────────────────────────────────────────────────────────────┐
│                           END-TO-END PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────────────┐  │
│  │   RAW X-RAY  │───▶│   PREPROCESSING  │───▶│   DATA AUGMENTATION     │  │
│  │    IMAGE     │    │  (Resize 224x224  │    │  (Rotation, Flip, Zoom) │  │
│  └──────────────┘    │   Normalization)  │    └─────────────────────────┘  │
│                      └──────────────────┘                                  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                     CLASSIFICATION TASK                               │ │
│  │  ┌─────────────────────────┐      ┌──────────────────────────────┐  │ │
│  │  │   BINARY CLASSIFICATION │      │  THREE-CLASS CLASSIFICATION  │  │ │
│  │  │      (ResNet-50)        │      │     (DenseNet-121)           │  │ │
│  │  │  ┌───────────────────┐  │      │  ┌────────────────────────┐  │  │ │
│  │  │  │  Global Avg Pool  │  │      │  │   Global Avg Pool      │  │  │ │
│  │  │  │  Dense (256,ReLU) │  │      │  │   Dense (256,ReLU)     │  │  │ │
│  │  │  │  Dropout (0.5)    │  │      │  │   Dropout (0.5)        │  │  │ │
│  │  │  │  Output (2,Sigmoid)│  │      │  │   Output (3,Softmax)   │  │  │ │
│  │  │  └───────────────────┘  │      │  └────────────────────────┘  │  │ │
│  │  └─────────────────────────┘      └──────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│  ┌─────────────────────────────────┴─────────────────────────────────────┐ │
│  │                     EXPLAINABILITY LAYER                              │ │
│  │  ┌─────────────────────────────┐      ┌──────────────────────────┐  │ │
│  │  │         GRAD-CAM            │      │          LIME            │  │ │
│  │  │  (Gradient-based,           │      │  (Model-agnostic,        │  │ │
│  │  │   Model-specific)           │      │   Perturbation-based)    │  │ │
│  │  │  ┌───────────────────────┐  │      │  ┌──────────────────────┐ │  │
│  │  │  │  Class Activation Map │  │      │  │  Local Surrogate     │ │  │
│  │  │  │  Heatmap Generation   │  │      │  │  Model Explanation   │ │  │
│  │  │  └───────────────────────┘  │      │  └──────────────────────┘ │  │
│  │  └─────────────────────────────┘      └──────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│  ┌─────────────────────────────────┴─────────────────────────────────────┐ │
│  │                     EVALUATION LAYER                                  │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │  │  Metrics: Clinician Agreement Score | Localization Accuracy   │ │ │
│  │  │          F1-Score | AUC-ROC | Precision | Recall             │ │ │
│  │  │  Validation: Stratified 5-Fold Cross-Validation              │ │ │
│  │  │  Statistical Tests: Paired t-test | McNemar's test           │ │ │
│  │  └─────────────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘



## Current Status
🔧 In Progress — Milestone 2: Preprocessing scripts and initial folder structure 
completed. Model training and XAI evaluation pipeline in development.

## Evaluation Metrics
- Accuracy, F1-score, AUC-ROC (classification)
- Clinician agreement score, localization accuracy (explanation quality)
- Stratified k-fold cross-validation
- Statistical significance testing (paired t-test / Wilcoxon signed-rank test)

 