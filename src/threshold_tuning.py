
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

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
        ("num", StandardScaler(), numerical_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ]
)

# Logistic Regression model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model.fit(X_train, y_train)

# Get probability of Bad Credit (class 0)
probabilities = model.predict_proba(X_test)[:, 0]

# Test different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6]

print("\n--- CLASSIFICATION THRESHOLD COMPARISON ---")

for threshold in thresholds:

    # Predict Bad Credit when its probability >= threshold
    predictions = (probabilities >= threshold).astype(int)

    # Convert predictions into original label format:
    # 0 = Bad Credit, 1 = Good Credit
    y_pred = 1 - predictions

    print(f"\nThreshold: {threshold}")

    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Bad Credit Precision:", round(
        precision_score(y_test, y_pred, pos_label=0, zero_division=0), 4
    ))
    print("Bad Credit Recall:", round(
        recall_score(y_test, y_pred, pos_label=0, zero_division=0), 4
    ))
    print("Bad Credit F1:", round(
        f1_score(y_test, y_pred, pos_label=0, zero_division=0), 4
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
