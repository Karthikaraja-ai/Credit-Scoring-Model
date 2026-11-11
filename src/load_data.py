
from pathlib import Path
from sklearn.datasets import fetch_openml

# Download the German Credit dataset
credit = fetch_openml(
    name="credit-g",
    version=1,
    as_frame=True
)

# Convert the dataset into a DataFrame
df = credit.frame

# Save the dataset
project_folder = Path(__file__).resolve().parents[1]
file_path = project_folder / "data" / "raw" / "credit_data.csv"

df.to_csv(file_path, index=False)

print("Dataset downloaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
