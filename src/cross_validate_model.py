from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Load dataset
project_folder = Path(__file__).resolve().parents[1]
file_path = project_folder / "data" / "raw" / "credit_data.csv"

df = pd.read_csv(file_path)

# Separate features and target
X = df.drop(columns=["class"])
y = df["class"].map({"good": 1, "bad": 0})

# Identify columns
numerical_columns = X.select_dtypes(include="number").columns
categorical_columns = X.select_dtypes(exclude="number").columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ]
)

# Random Forest model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    ))
])

# Stratified 5-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Evaluate model
scoring = {
    "accuracy": "accuracy",
    "bad_credit_precision": make_scorer(
        precision_score, pos_label=0, zero_division=0
    ),
    "bad_credit_recall": make_scorer(
        recall_score, pos_label=0, zero_division=0
    ),
    "bad_credit_f1": make_scorer(
        f1_score, pos_label=0, zero_division=0
    ),
    "roc_auc": "roc_auc"
}
results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring
)

# Display results
print("\n--- 5-Fold Cross-Validation Results ---")

for metric in scoring:
    scores = results[f"test_{metric}"]

    print(f"\n{metric.upper()}")
    print("Individual scores:", scores.round(4))
    print("Mean score:", round(scores.mean(), 4))
    print("Standard deviation:", round(scores.std(), 4))
