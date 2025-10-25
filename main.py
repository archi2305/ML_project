import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ----------------------------
# 1️⃣ File Setup
# ----------------------------
output_folder = 'processed_data'
file_name = "data/Extended_Employee_Performance_and_Productivity_Data.csv"

# ----------------------------
# 2️⃣ Load Dataset
# ----------------------------
try:
    df = pd.read_csv(file_name)
    print("✅ Dataset loaded successfully!")
    print(f"Original shape: {df.shape}")
except FileNotFoundError:
    print("❌ ERROR: File not found!")
    exit()

# ----------------------------
# 3️⃣ Remove Irrelevant Columns
# ----------------------------
irrelevant_cols = ['Employee_ID', 'Job_Title', 'Hire_Date']
df_cleaned = df.drop(columns=irrelevant_cols, errors='ignore')
print(f"\nRemoved irrelevant columns: {irrelevant_cols}")
print(f"Shape after cleaning: {df_cleaned.shape}")

# ----------------------------
# 4️⃣ Handle Missing Values
# ----------------------------
print(f"Missing values before cleaning: {df_cleaned.isnull().sum().sum()}")
df_cleaned = df_cleaned.dropna()
print(f"Missing values after cleaning: {df_cleaned.isnull().sum().sum()}")

# ----------------------------
# 5️⃣ Encode Categorical Columns
# ----------------------------
text_features = ['Department', 'Gender', 'Education_Level', 'Resigned']
df_processed = pd.get_dummies(df_cleaned, columns=text_features, drop_first=True)
print(f"\n✅ Encoding done! New shape: {df_processed.shape}")

# ----------------------------
# 6️⃣ Split Features and Target
# ----------------------------
y = df_processed['Performance_Score']
X = df_processed.drop(columns=['Performance_Score'])
print(f"X shape: {X.shape}, y shape: {y.shape}")

# ----------------------------
# 7️⃣ Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n✅ Data split into train and test sets!")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ----------------------------
# 8️⃣ Scale Data
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 9️⃣ Save Processed Files
# ----------------------------
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created folder: {output_folder}")

# Save arrays and targets
np.save(os.path.join(output_folder, 'X_train_scaled.npy'), X_train_scaled)
np.save(os.path.join(output_folder, 'X_test_scaled.npy'), X_test_scaled)
y_train.to_csv(os.path.join(output_folder, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(output_folder, 'y_test.csv'), index=False)

# Save scaler and feature columns
joblib.dump(scaler, os.path.join(output_folder, 'scaler.joblib'))
joblib.dump(X.columns, os.path.join(output_folder, 'columns.joblib'))

print("\n✅ All processed files saved successfully!")
print(f"- {output_folder}/X_train_scaled.npy")
print(f"- {output_folder}/X_test_scaled.npy")
print(f"- {output_folder}/y_train.csv")
print(f"- {output_folder}/y_test.csv")
print(f"- {output_folder}/scaler.joblib")
print(f"- {output_folder}/columns.joblib")
