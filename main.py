# ---------------------------------------------
# Corporate Resource Optimization - ML Pipeline (Final Stable Version)
# ---------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
import os

# ---------------- PATHS ----------------
DATA_PATH = "data/Extended_Employee_Performance_and_Productivity_Data.csv"
OUTPUT_FOLDER = "processed_data"
MODEL_FOLDER = "models"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)
print("✅ Dataset loaded successfully:", df.shape)
print("\n🔍 Columns in dataset:\n", df.columns.tolist())

# Drop irrelevant columns
irrelevant_cols = ['Employee_ID', 'Job_Title', 'Hire_Date']
df_cleaned = df.drop(columns=irrelevant_cols, errors='ignore')

# Handle missing data
df_cleaned.dropna(inplace=True)
print("✅ Data cleaned:", df_cleaned.shape)

# ---------------- SELECT CORRECT COLUMNS ----------------
# Based on your dataset, these columns exist:
hours_col = 'Work_Hours_Per_Week'
budget_col = 'Monthly_Salary'

# ---------------- FEATURE ENGINEERING ----------------
df_cleaned['Efficiency'] = df_cleaned[hours_col] / (df_cleaned[budget_col] + 0.1)
df_cleaned['WorkBudgetRatio'] = df_cleaned[hours_col] * df_cleaned[budget_col]
df_cleaned['EngagementScore'] = (
    df_cleaned['Projects_Handled'] +
    df_cleaned['Training_Hours'] -
    df_cleaned['Sick_Days'] +
    df_cleaned['Overtime_Hours']
)
print("\n✅ Feature Engineering Added: ['Efficiency', 'WorkBudgetRatio', 'EngagementScore']")

# ---------------- ENCODE CATEGORICAL FEATURES ----------------
text_features = ['Department', 'Gender', 'Education_Level', 'Resigned']
df_processed = pd.get_dummies(df_cleaned, columns=text_features, drop_first=True)
print("\n✅ Encoding done. Columns:")
print(df_processed.columns)

print("\nDistribution of Performance_Score:")
print(df_processed['Performance_Score'].value_counts())

# ---------------- SPLIT FEATURES & TARGET ----------------
y = df_processed['Performance_Score']
X = df_processed.drop(columns=['Performance_Score'])

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# ---------------- SCALE FEATURES ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- TRAIN MODEL (RandomForest Classifier) ----------------
print("\n🚀 Training RandomForest model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("✅ Model training complete!")

# ---------------- EVALUATE MODEL ----------------
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {round(accuracy*100,2)}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Performance Score Prediction - Confusion Matrix")
plt.show()

# ---------------- SAVE OUTPUTS ----------------
joblib.dump(model, os.path.join(MODEL_FOLDER, "Decision_Tree_model.pkl"))
joblib.dump(scaler, os.path.join(OUTPUT_FOLDER, "scaler.joblib"))
joblib.dump(X.columns.tolist(), os.path.join(OUTPUT_FOLDER, "columns.joblib"))

np.save(os.path.join(OUTPUT_FOLDER, 'X_train_scaled.npy'), X_train_scaled)
np.save(os.path.join(OUTPUT_FOLDER, 'X_test_scaled.npy'), X_test_scaled)
y_train.to_csv(os.path.join(OUTPUT_FOLDER, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(OUTPUT_FOLDER, 'y_test.csv'), index=False)

print("\n💾 Files saved successfully!")
print(f"Model Path: {MODEL_FOLDER}/Decision_Tree_model.pkl")
print(f"Scaler Path: {OUTPUT_FOLDER}/scaler.joblib")
print(f"Columns Path: {OUTPUT_FOLDER}/columns.joblib")

print("\n🎯 Model ready for integration with Flask API and Streamlit Dashboard!")
