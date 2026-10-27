# Child Malnutrition Classification System

A comprehensive machine learning and geometric morphometrics (GM) pipeline for child malnutrition classification (SAM, MAM, MC vs ONC) developed for the Andhra Pradesh Government project.

## Project Overview

This repository contains end-to-end data pipelines, exploratory data analysis (EDA), data augmentation techniques, model training workflows, and deployment applications for diagnosing childhood malnutrition:

- **MC vs ONC Classification**: Distinguishing Malnourished Children (MC - combining SAM & MAM) from Optimally Nourished Children (ONC).
- **Geometric Morphometrics (GM)**: Procrustes alignment, shape variation analysis, and out-of-sample shape classification.
- **Data Augmentation**: Geometric transformation modeling, SMOTE, and Thin Plate Splines (TPS) for balancing clinical cohorts.
- **Interactive Full Demo**: Production deployment application (`MC-ONC-Based-Classification/Full_Demo_MC_ONC_Classification`) featuring MediaPipe pose landmarking, ensemble classifiers (LDA, XGBoost, EGAT), and diagnostic reporting.

## Repository Structure

```text
├── .gitignore
├── requirements.txt
├── README.md
├── GM_out_of_sample_classification-ap-govt/  # AP Govt EDA, augmentation & training pipelines
├── GM_out_of_sample_classification-main/     # Core GM out-of-sample classification benchmarks
├── MC-ONC-Based-Classification/              # Production pipelines, models, and Full Demo app
├── SAM-Photo/                                # Photo diagnosis, WHO Anthro validation & reports
├── data filteration_test/                    # Data filtering and validation experiments
└── scratch/                                  # Development scripts and utilities
```

## Getting Started

### 1. Environment Setup
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configuration
For the demo app in `MC-ONC-Based-Classification/Full_Demo_MC_ONC_Classification`:
```bash
# Copy template and fill credentials if needed
copy .env.example .env
```

### 3. Running the Demo Application
```bash
cd MC-ONC-Based-Classification/Full_Demo_MC_ONC_Classification
python run_demo.py
```