
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

project_folder = Path(__file__).resolve().parents[1]
file_path = project_folder / "data" / "raw" / "credit_data.csv"

df = pd.read_csv(file_path)

sns.set_theme(style="whitegrid")

# Create one window with two graphs
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Graph 1: Credit Risk Distribution
sns.countplot(data=df, x="class", ax=axes[0])
axes[0].set_title("Credit Risk Distribution")
axes[0].set_xlabel("Credit Risk")
axes[0].set_ylabel("Number of Applicants")

# Graph 2: Loan Duration Distribution
sns.histplot(data=df, x="duration", bins=20, kde=True, ax=axes[1])
axes[1].set_title("Loan Duration Distribution")
axes[1].set_xlabel("Loan Duration (Months)")
axes[1].set_ylabel("Number of Applicants")

plt.tight_layout()
plt.show()
