
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Project paths
project_folder = Path(__file__).resolve().parents[1]
data_path = project_folder / "data" / "raw" / "credit_data.csv"
model_path = project_folder / "models" / "credit_scoring_model.pkl"

# Create models folder
model_path.parent.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(data_path)

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

# Model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, model_path)

print("Model saved successfully!")
print("Saved location:", model_path)
