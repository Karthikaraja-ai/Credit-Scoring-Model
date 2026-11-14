# Credit Scoring Model

## Overview
A Machine Learning project that predicts an applicant's creditworthiness using personal and financial information from the German Credit Dataset.

## Objectives
- Explore and understand credit data.
- Perform data preprocessing and feature engineering.
- Train and compare classification models.
- Evaluate models using multiple performance metrics.
- Save and reuse trained models for predictions.

## Dataset
- **Name:** German Credit Dataset
- **Records:** 1,000
- **Target:** Good Credit / Bad Credit

## Machine Learning Models
1. Logistic Regression
2. Decision Tree
3. Random Forest

## Data Preprocessing
- StandardScaler for numerical features.
- OneHotEncoder for categorical features.
- Stratified data splitting.
- Pipeline-based preprocessing and model training.

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Stratified 5-Fold Cross-Validation

## Model Comparison Results

All three models were evaluated using Stratified 5-Fold Cross-Validation.

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 71.9% | 78.55% |
| Decision Tree | 63.6% | 67.18% |
| Random Forest | 72.1% | 79.21% |

## Random Forest — Cross-Validation Details

| Metric | Mean Score | Standard Deviation |
|---|---:|---:|
| Accuracy | 72.10% | 2.44% |
| Bad Credit Precision | 52.94% | 3.43% |
| Bad Credit Recall | 67.67% | 1.33% |
| Bad Credit F1-Score | 59.34% | 2.23% |
| ROC-AUC | 79.21% | 2.12% |

The results indicate relatively consistent performance across the five validation folds.

## Additional Experiments
- Classification threshold tuning.
- Model saving and loading using Joblib.
- Sample applicant predictions with probability scores.

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
│   ├── raw/
│   └── processed/
├── models/
│   └── credit_scoring_model.pkl
├── reports/
│   └── model_comparison.md
├── src/
│   ├── explore_data.py
│   ├── preprocess_data.py
│   ├── train_model.py
│   ├── train_decision_tree.py
│   ├── train_random_forest.py
│   ├── compare_models.py
│   ├── cross_validate_model.py
│   ├── threshold_tuning.py
│   ├── save_model.py
│   └── load_model.py
├── README.md
├── project_notes.md
├── requirements.txt
└── test_setup.py
```

## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/Karthikaraja-ai/Credit-Scoring-Model.git
cd Credit-Scoring-Model
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the Project

Explore the dataset:

```bash
python src/explore_data.py
```

Compare machine learning models:

```bash
python src/compare_models.py
```

Perform cross-validation:

```bash
python src/cross_validate_model.py
```

Tune classification thresholds:

```bash
python src/threshold_tuning.py
```

Save and load the trained model:

```bash
python src/save_model.py
python src/load_model.py
```

## Limitations
This is an educational project. Its results should not be used for actual lending decisions without further validation, fairness testing, and appropriate safeguards.

## Author
Machine Learning Internship Project — CodeAlpha