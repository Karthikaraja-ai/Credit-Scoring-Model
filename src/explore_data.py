import pandas as pd
from pathlib import Path

# Locate and load the dataset
project_folder = Path(__file__).resolve().parents[1]
file_path = project_folder / "data" / "raw" / "credit_data.csv"

df = pd.read_csv(file_path)

# 1. Dataset dimensions
print("Dataset Shape:")
print(df.shape)

# 2. First five records
print("\nFirst 5 Records:")
print(df.head())

# 3. Column information
print("\nDataset Information:")
df.info()

# 4. Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 5. Target distribution
print("\nCredit Risk Distribution:")
print(df["class"].value_counts())
