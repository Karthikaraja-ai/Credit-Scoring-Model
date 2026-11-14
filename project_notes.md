# Credit Scoring Model

## Objective
Predict whether a loan applicant is a good or bad credit risk using machine learning classification algorithms.

## Machine Learning Type
Supervised Learning

## Problem Type
Binary Classification

## Dataset
German Credit Dataset
- Total records: 1,000
- Target classes: Good Credit and Bad Credit

## Features
Personal and financial information of loan applicants, including numerical and categorical attributes.

## Target
Credit Risk:
- Good Credit (1)
- Bad Credit (0)

## Algorithms
1. Logistic Regression
2. Decision Tree
3. Random Forest

## Data Preprocessing
- Numerical feature scaling using StandardScaler
- Categorical feature encoding using OneHotEncoder
- Stratified train-test splitting
- Unknown categorical values handled using `handle_unknown="ignore"`

## Model Evaluation
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Stratified 5-Fold Cross-Validation

## Additional Experiments
- Comparison of three classification algorithms
- Cross-validation of the Random Forest model
- Classification threshold tuning
- Model saving and loading using Joblib
- Sample applicant predictions with probability scores

## Best Model Comparison Results
Random Forest:
- Accuracy: 72.1%
- Bad Credit Precision: 52.94%
- Bad Credit Recall: 67.67%
- Bad Credit F1-Score: 59.34%
- ROC-AUC: 79.21%

These are mean 5-fold cross-validation results from the model comparison experiment.

## Limitations
- The dataset contains only 1,000 records.
- Model performance may not generalize to real-world lending applications.
- Further validation and fairness analysis are necessary before any real-world use.
- This project is developed for educational purposes.