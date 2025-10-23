import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Path to dataset
DATA_PATH = os.path.join("data", "Extended_Employee_Performance_and_Productivity_Data.csv")

try:
	df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
	print(f"CSV not found at {DATA_PATH}", file=sys.stderr)
	sys.exit(1)
except Exception as e:
	print("Error reading CSV:", e, file=sys.stderr)
	sys.exit(1)

# Display first few rows
print(df.head())

# Check info about columns (note: df.info() prints and returns None)
print(df.info())

# Summary statistics
print(df.describe())

# Correct column name (remove accidental trailing comma in original code)
col = "Performance_Score"
if col not in df.columns:
	print(f"Column '{col}' not found. Available columns: {df.columns.tolist()}", file=sys.stderr)
else:
	sns.countplot(x=col, data=df)
	plt.title(col)
	plt.show()
