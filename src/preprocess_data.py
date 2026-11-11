import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parents[1]
file_path = project_folder / "data" / "raw" / "credit_data.csv"

df = pd.read_csv(file_path)

# Identify numerical columns
numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

# Identify categorical columns
categorical_columns = df.select_dtypes(
    include=["object","str", "category"]
).columns.tolist()

print("Numerical Columns:")
print(numerical_columns)

print("\nCategorical Columns:")
print(categorical_columns)

print("\nTarget Column:")
print(df["class"].value_counts())
