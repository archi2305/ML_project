from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

# -----------------------------
# 1️⃣ INITIAL SETUP
# -----------------------------
app = Flask(__name__)

# Dynamically set absolute base directory (works for macOS)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define absolute file paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "Decision_Tree_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "processed_data", "scaler.joblib")
COLUMNS_PATH = os.path.join(BASE_DIR, "processed_data", "columns.joblib")

print("📂 Base Directory:", BASE_DIR)
print("🧠 Model Path:", MODEL_PATH)
print("⚙️ Scaler Path:", SCALER_PATH)
print("📊 Columns Path:", COLUMNS_PATH)

# -----------------------------
# 2️⃣ LOAD MODEL, SCALER, AND COLUMNS
# -----------------------------
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

try:
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler Loaded Successfully!")
except Exception as e:
    print("❌ Error loading scaler:", e)
    scaler = None

try:
    columns = joblib.load(COLUMNS_PATH)
    print("✅ Columns Loaded Successfully!")
except Exception as e:
    print("❌ Error loading columns:", e)
    columns = None

if (model is not None) and (scaler is not None) and (columns is not None):
    print("🎯 All files loaded successfully and ready for prediction!")
else:
    print("⚠️ One or more files failed to load — please check above messages.")

# -----------------------------
# 3️⃣ HOME ROUTE
# -----------------------------
@app.route('/')
def home():
    return jsonify({
        "message": "🚀 Corporate Resource Optimization API is live!",
        "status": "success"
    })

# -----------------------------
# 4️⃣ PREDICTION ROUTE
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No input data received"})

        if model is None or scaler is None or columns is None:
            return jsonify({
                "status": "error",
                "message": "Model, scaler, or columns not loaded properly."
            })

        # Convert input JSON to DataFrame
        df = pd.DataFrame([data])
        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)

        # Scale data
        scaled_input = scaler.transform(df)

        # Predict
        prediction = model.predict(scaled_input)[0]

        # Map numeric output to label
        performance_labels = {
            1: "Low",
            2: "Average",
            3: "Good",
            4: "Excellent",
            5: "Outstanding"
        }
        predicted_label = performance_labels.get(int(prediction), "Unknown")

        print(f"✅ Input: {data} → Prediction: {prediction} ({predicted_label})")

        return jsonify({
            "status": "success",
            "prediction": float(prediction),
            "performance_label": predicted_label
        })

    except Exception as e:
        print("❌ Error in /predict route:", e)
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# -----------------------------
# 5️⃣ RUN FLASK APP
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)
