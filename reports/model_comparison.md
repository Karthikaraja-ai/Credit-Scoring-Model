
# Credit Scoring Model — Model Comparison

## 5-Fold Cross-Validation Results

| Metric | Logistic Regression | Decision Tree | Random Forest |
|---|---:|---:|---:|
| Accuracy | 71.9% | 63.6% | 72.1% |
| Bad Credit Precision | 52.39% | 43.46% | 52.94% |
| Bad Credit Recall | 72.00% | 65.67% | 67.67% |
| Bad Credit F1 | 60.61% | 51.92% | 59.34% |
| ROC-AUC | 78.55% | 67.18% | 79.21% |

## Observations

- Logistic Regression achieved the highest bad-credit recall and F1-score.
- Random Forest achieved the highest accuracy, bad-credit precision and ROC-AUC.
- Decision Tree performed less effectively than the other two models.

## Conclusion

Logistic Regression and Random Forest are the strongest candidates.

Logistic Regression is preferable when identifying more potentially risky applicants is the priority.

Random Forest provides slightly better overall discrimination based on ROC-AUC.

The final model choice depends on the cost of false approvals and false rejections.

## Limitations

- The German Credit dataset contains only 1,000 records.
- Results may not generalize to current real-world lending populations.
- Further fairness, bias and external validation are required before practical deployment.
