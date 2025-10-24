import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import sys

output_folder = 'processed_data'
model_folder="models"
data_file= "data/Extended_Employee_Performance_and_Productivity_Data.csv" 

try:
    df = pd.read_csv(data_file)
    print("Dataset loaded successfully!")
    print(f"Original shape: {df.shape}")
    
except FileNotFoundError:
    print(f"--- ERROR: FILE NOT FOUND ---")
    sys.exit()
    
irrelevant_cols = ['Employee_ID', 'Job_Title', 'Hire_Date']#not necessary for training and testing of data ,these are noise

# We will create a new, clean dataframe
df_cleaned = df.drop(columns=irrelevant_cols,errors="ignore")

print("\n--- Removed irrelevant columns ---")
print(f"New shape: {df_cleaned.shape}")
print("Remaining columns:", list(df_cleaned.columns))

# Check for missing values BEFORE
print(f"\n--- Total missing values before cleaning: {df_cleaned.isnull().sum().sum()} ---")

# Drop any row with one or more 'NA' (missing) values
df_cleaned = df_cleaned.dropna()

# Check for missing values AFTER
print(f"--- Total missing values after cleaning: {df_cleaned.isnull().sum().sum()} ---")
print(f"Shape after dropping missing rows: {df_cleaned.shape}")

# These are the text columns we need to convert
text_features = ['Department', 'Gender', 'Education_Level', 'Resigned']


df_processed = pd.get_dummies(df_cleaned, columns=text_features, drop_first=True)


print(df_processed.head(3))
# Save the column structure for future use (Flask needs it)
joblib.dump(df_processed.columns, os.path.join(output_folder, "columns.joblib"))


if 'Performance_Score' not in df_processed.columns:
    print("❌ ERROR: 'Performance_Score' column not found. Please check your dataset.")
    sys.exit()
y = df_processed['Performance_Score']
X = df_processed.drop(columns=['Performance_Score'])

print("\n--- Separated data into Features (X) and Target (y) ---")
print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Split data into 80% training and 20% testing ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

from sklearn.preprocessing import StandardScaler

# 1. Initialize the scaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)





if not os.path.exists(output_folder):
    os.makedirs(output_folder)

np.save(os.path.join(output_folder, 'X_train_scaled.npy'), X_train_scaled)
np.save(os.path.join(output_folder, 'X_test_scaled.npy'), X_test_scaled)
y_train.to_csv(os.path.join(output_folder, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(output_folder, 'y_test.csv'), index=False)
joblib.dump(scaler, os.path.join(output_folder, 'scaler.joblib'))

print("\n✅ Saved processed data to folder:", output_folder)



dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)

# 2️⃣ Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

# --- Step 10: Model Evaluation ---
y_pred_dt = dt_model.predict(X_test_scaled)
y_pred_lr = lr_model.predict(X_test_scaled)

print("\n🔹 DECISION TREE PERFORMANCE 🔹")
print("Accuracy:", round(accuracy_score(y_test, y_pred_dt), 3))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))

print("\n🔹 LOGISTIC REGRESSION PERFORMANCE 🔹")
print("Accuracy:", round(accuracy_score(y_test, y_pred_lr), 3))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))


acc_dt = accuracy_score(y_test, y_pred_dt)
acc_lr = accuracy_score(y_test, y_pred_lr)

best_model = dt_model if acc_dt > acc_lr else lr_model
best_model_name = "Decision Tree" if acc_dt > acc_lr else "Logistic Regression"
print(f"\n🏆 Best Model: {best_model_name} with Accuracy = {round(max(acc_dt, acc_lr)*100, 2)}%")


if not os.path.exists(model_folder):
    os.makedirs(model_folder)

model_path = os.path.join(model_folder, f"{best_model_name.replace(' ', '_')}_model.pkl")
joblib.dump(best_model, model_path)

print(f"\n✅ Model saved successfully at: {model_path}")


print("\n===============================")
print("🏁 DATA PREPROCESSING & TRAINING COMPLETED")
print(f"Total rows used: {len(df_processed)}")
print(f"Best Model: {best_model_name}")
print(f"Accuracy: {round(max(acc_dt, acc_lr)*100, 2)}%")
print("===============================")
