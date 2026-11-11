
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split

# Project paths
project_folder = Path(__file__).resolve().parents[1]
data_path = project_folder / "data" / "raw" / "credit_data.csv"
model_path = project_folder / "models" / "credit_scoring_model.pkl"

# Load saved model
model = joblib.load(model_path)

print("Model loaded successfully!")

# Load data for testing predictions
df = pd.read_csv(data_path)

X = df.drop(columns=["class"])
y = df["class"].map({"good": 1, "bad": 0})

_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Select five test records
sample_data = X_test.head(5)
actual_results = y_test.head(5)

# Make predictions
predictions = model.predict(sample_data)
probabilities = model.predict_proba(sample_data)

# Display results
for i in range(len(sample_data)):
    actual = "Good Credit" if actual_results.iloc[i] == 1 else "Bad Credit"
    predicted = "Good Credit" if predictions[i] == 1 else "Bad Credit"

    print(f"\nApplicant {i + 1}")
    print("Actual:", actual)
    print("Predicted:", predicted)
    print(f"Bad Credit Probability: {probabilities[i][0]:.2%}")
    print(f"Good Credit Probability: {probabilities[i][1]:.2%}")
