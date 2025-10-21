import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

output_folder = 'processed_data'
# --- Make sure your CSV file is in your 'data' folder ---
# --- !! UPDATE THIS FILENAME to match your file !! ---
# Based on your screenshot, the file might be named this:
file_name = "data/Extended_Employee_Performance_and_Productivity_Data.csv" 

try:
    df = pd.read_csv(file_name)
    print("--- Dataset loaded successfully! ---")
    print(f"Original shape: {df.shape}")
    print(df.head())
except FileNotFoundError:
    print(f"--- ERROR: FILE NOT FOUND ---")
    print(f"I tried to find '{file_name}' but couldn't.")
    print("Please check the name of your file in the 'data' folder and update the 'file_name' variable.")

    # These columns are noise (ID) or redundant (Hire_Date) or too complex for now (Job_Title)
irrelevant_cols = ['Employee_ID', 'Job_Title', 'Hire_Date']

# We will create a new, clean dataframe
df_cleaned = df.drop(columns=irrelevant_cols)

print("\n--- Removed irrelevant columns ---")
print(f"New shape: {df_cleaned.shape}")
print("Remaining columns:", df_cleaned.columns)

# Check for missing values BEFORE
print(f"\n--- Total missing values before cleaning: {df_cleaned.isnull().sum().sum()} ---")

# Drop any row with one or more 'NA' (missing) values
df_cleaned = df_cleaned.dropna()

# Check for missing values AFTER
print(f"--- Total missing values after cleaning: {df_cleaned.isnull().sum().sum()} ---")
print(f"Shape after dropping missing rows: {df_cleaned.shape}")

# These are the text columns we need to convert
text_features = ['Department', 'Gender', 'Education_Level', 'Resigned']

# We use 'pd.get_dummies' to do One-Hot Encoding
# This will create new columns (e.g., 'Department_HR', 'Department_IT', 'Gender_Male', etc.)
# drop_first=True is good practice to prevent a math issue called multicollinearity
df_processed = pd.get_dummies(df_cleaned, columns=text_features, drop_first=True)

print("\n--- Text features encoded successfully ---")
print("Your data is now 100% numbers.")
print(df_processed.head())

# Our target 'y' is the column we want to predict
y = df_processed['Performance_Score']

# Our features 'X' are ALL other columns
# We drop the target column from our feature set
X = df_processed.drop(columns=['Performance_Score'])

print("\n--- Separated data into Features (X) and Target (y) ---")
print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

from sklearn.model_selection import train_test_split

# test_size=0.2 means 20% for testing
# random_state=42 ensures you get the same "random" split every time you run the code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Split data into 80% training and 20% testing ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

from sklearn.preprocessing import StandardScaler

# 1. Initialize the scaler
scaler = StandardScaler()

# 2. Fit the scaler ONLY on the training data
scaler.fit(X_train)

# 3. Transform both the train and test data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Data scaled successfully! ---")
print("All features are now on the same scale.")
print("Example of scaled data (first row of X_train_scaled):")
print(X_train_scaled[0])

# ==========================================================
# PART 3: SAVE PROCESSED DATA FOR DAY 3
# ==========================================================
# --- Create the output folder if it doesn't exist ---
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created new folder: {output_folder}")
print(f"\n[Step 8] Saving processed data to '{output_folder}/' folder...")

# Save the scaled (NumPy) arrays
np.save(os.path.join(output_folder, 'X_train_scaled.npy'), X_train_scaled)
np.save(os.path.join(output_folder, 'X_test_scaled.npy'), X_test_scaled)

# Save the target (Pandas Series) as CSVs
y_train.to_csv(os.path.join(output_folder, 'y_train.csv'), index=False, header=True)
y_test.to_csv(os.path.join(output_folder, 'y_test.csv'), index=False, header=True)

# Save the scaler object itself for later use
scaler_path = os.path.join(output_folder, 'scaler.joblib')
joblib.dump(scaler, scaler_path)

print("All processed files saved successfully:")
print(f"- {output_folder}/X_train_scaled.npy")
print(f"- {output_folder}/X_test_scaled.npy")
print(f"- {output_folder}/y_train.csv")
print(f"- {output_folder}/y_test.csv")
print(f"- {output_folder}/scaler.joblib")

print("\n--- Day 2 Script FINISHED Successfully! ---")
print("You are ready for Day 3 (Model Training).")