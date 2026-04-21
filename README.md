# smart-helmet-ai-module

AI/ML module for the Smart Safety Helmet final year project.

This repository contains the model logic and training pipeline used to:

- detect abnormal sensor behavior,
- classify fall-related motion patterns,
- estimate near-future hazard risk from sensor trends.

## Project Structure

- `ai_models.py` - core AI engine and model implementations
- `train_models.py` - model training and evaluation script
- `models/` - saved model artifacts (`.joblib`)
- `output/` - generated evaluation reports

## Models Included

### 1) Anomaly Detection

Uses `IsolationForest` to learn normal sensor patterns and flag unusual behavior.

### 2) Fall Classification

Uses `RandomForestClassifier` to classify motion states:

- normal
- walking
- fall
- aftermath

### 3) Hazard Prediction

Uses trend-based analysis and regression to estimate near-term risk escalation.

## Requirements

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install numpy pandas scikit-learn joblib
```

## Training

From the repository root:

```bash
python train_models.py
```

Training outputs:

- `models/anomaly_detector.joblib`
- `models/fall_classifier.joblib`
- `output/ai_evaluation.json`

## Notes

- `train_models.py` expects simulator outputs in `../simulator/output`.
- Ensure simulation files exist before starting training.

## Authors

- Jean D Amour KUBWIMANA
- Jean de Dieu NIYONKURU

University of Rwanda - Final Year Project (2026)
