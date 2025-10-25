# ---------------------------------------------
# Corporate Resource Optimization - Flask API
# ---------------------------------------------
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# ---------------- LOAD SAVED ARTIFACTS ----------------
MODEL_PATH = "../models/Decision_Tree_model.pkl"
SCALER_PATH = "../processed_data/scaler.joblib"
COLUMNS_PATH = "../processed_data/columns.joblib"

model = None
scaler = None
columns = None

try:
    print("✅ Loading model, scaler, and columns...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    columns = joblib.load(COLUMNS_PATH)
    print("✅ Model, Scaler, and Columns Loaded Successfully!")
except Exception as e:
    print("❌ Error loading model files:", e)


# ---------------- API ROUTES ----------------
@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to Corporate Resource Optimization API",
        "status": "running"
    })


@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler, columns

    if model is None or scaler is None or columns is None:
        return jsonify({"error": "Model or scaler not loaded properly"}), 500

    try:
        # Input data (JSON)
        data = request.get_json(force=True)
        df = pd.DataFrame([data])

        # Ensure all expected columns exist
        df_processed = pd.get_dummies(df)
        for col in columns:
            if col not in df_processed.columns:
                df_processed[col] = 0
        df_processed = df_processed[columns]

        # Scale input
        scaled_data = scaler.transform(df_processed)

        # Predict
        prediction = model.predict(scaled_data)[0]

        return jsonify({
            "prediction": int(prediction),
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        })


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
