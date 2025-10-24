# ============================
# Corporate Resource Optimization Dashboard (Backend)
# Flask + Machine Learning Integration
# ============================

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import traceback

# Initialize Flask app
app = Flask(__name__)

# --- Load Model, Scaler, and Column Structure ---
try:
    model = joblib.load("../models/Decision_Tree_model.pkl")
    scaler = joblib.load("../processed_data/scaler.joblib")
    columns = joblib.load("../processed_data/columns.joblib")
    print("✅ Model, Scaler, and Columns Loaded Successfully!")
except Exception as e:
    print("❌ Error loading model or scaler:", e)
    traceback.print_exc()

# ============================
# ROUTES
# ============================

@app.route('/')
def home():
    """Default route - to confirm backend is running"""
    return "<h2>🚀 Corporate Resource Optimization Model is Running!</h2>"

# ----------------------------
# PREDICTION ROUTE
# ----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input data
        data = request.get_json()

        # Convert input data into a DataFrame
        input_df = pd.DataFrame([data])

        # Apply same encoding and structure as training
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=columns, fill_value=0)

        # Apply the saved scaler
        input_scaled = scaler.transform(input_df)

        # Make prediction using the trained model
        prediction = model.predict(input_scaled)[0]

        return jsonify({
            "status": "success",
            "prediction": str(prediction)
        })

    except Exception as e:
        # Handle errors gracefully
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# ----------------------------
# SERVER RUN CONFIG
# ----------------------------
if __name__ == '__main__':
    app.run(debug=True)
