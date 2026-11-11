
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

# Load dataset
project_folder = Path(__file__).resolve().parents[1]
file_path = project_folder / "data" / "raw" / "credit_data.csv"

df = pd.read_csv(file_path)

X = df.drop(columns=["class"])
y = df["class"].map({"good": 1, "bad": 0})

# Identify columns
numerical_columns = X.select_dtypes(include="number").columns
categorical_columns = X.select_dtypes(exclude="number").columns

# Cross-validation setup
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Evaluation metrics
scoring = {
    "accuracy": "accuracy",
    "bad_precision": make_scorer(
        precision_score, pos_label=0, zero_division=0
    ),
    "bad_recall": make_scorer(
        recall_score, pos_label=0, zero_division=0
    ),
    "bad_f1": make_scorer(
        f1_score, pos_label=0, zero_division=0
    ),
    "roc_auc": "roc_auc"
}

# Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )
}

results_list = []

# Train and evaluate each model
for name, classifier in models.items():

    # Scale numerical features for Logistic Regression
    if name == "Logistic Regression":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_columns),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
            ]
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", numerical_columns),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
            ]
        )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring
    )

    results_list.append({
        "Model": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Bad Precision": scores["test_bad_precision"].mean(),
        "Bad Recall": scores["test_bad_recall"].mean(),
        "Bad F1": scores["test_bad_f1"].mean(),
        "ROC-AUC": scores["test_roc_auc"].mean()
    })

# Display comparison
results_df = pd.DataFrame(results_list)

print("\n--- MODEL COMPARISON RESULTS ---")
print(results_df.round(4).to_string(index=False))
