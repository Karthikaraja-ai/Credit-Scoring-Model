# Credit Scoring Model

## Overview
A Machine Learning project that predicts an applicant's creditworthiness using financial and personal information from the German Credit dataset.

## Objectives
- Understand and explore credit data.
- Perform data preprocessing and feature engineering.
- Train and compare classification models.
- Evaluate models using multiple performance metrics.
- Save and reuse a trained model for predictions.

## Dataset
- **Name:** German Credit Dataset
- **Records:** 1,000
- **Target:** Good Credit / Bad Credit

## Machine Learning Models
1. Logistic Regression
2. Decision Tree
3. Random Forest

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- 5-Fold Stratified Cross-Validation

## Key Results

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 71.9% | 78.55% |
| Decision Tree | 63.6% | 67.18% |
| Random Forest | 72.1% | 79.21% |

*Results are from 5-fold cross-validation.*

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Project Structure
```text
Credit-Scoring-Model/
├── data/
│   └── raw/
├── models/
├── reports/
│   └── model_comparison.md
├── src/
├── README.md
└── requirements.txt
```

## Limitations
This is an educational project. Its results should not be used for actual lending decisions without further validation, fairness testing and appropriate safeguards.

## Author
Machine Learning Internship Project — CodeAlpha