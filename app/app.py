import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------------------------------------
# Define Flask app
# ---------------------------------------
app = Flask(__name__)
CORS(app)

# Get absolute base directory (so paths work no matter where it's run from)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Paths for model, scaler, and columns
MODEL_PATH = os.path.join(BASE_DIR, "models", "Decision_Tree_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "processed_data", "scaler.joblib")
COLUMNS_PATH = os.path.join(BASE_DIR, "processed_data", "columns.joblib")

# ---------------------------------------
# Load model, scaler, and columns safely
# ---------------------------------------
model, scaler, columns = None, None, None

try:
    print(f"📂 Loading model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")

    print(f"📂 Loading scaler from: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler loaded successfully!")

    print(f"📂 Loading columns from: {COLUMNS_PATH}")
    columns = joblib.load(COLUMNS_PATH)
    print("✅ Columns loaded successfully!")

except Exception as e:
    print(f"❌ Error loading model/scaler/columns: {e}")

# ---------------------------------------
# Prediction Endpoint
# ---------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler, columns
    if model is None or scaler is None or columns is None:
        return jsonify({
            "status": "error",
            "message": "Model or scaler not loaded properly."
        })

    try:
        data = request.get_json()
        print("📩 Received data:", data)

        # Validate incoming JSON payload
        if not data or not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Invalid or empty JSON payload"
            }), 400

        # Convert input JSON to array in correct column order.
        # columns contains final feature names (including one-hot columns)
        import pandas as pd

        # Initialize input series with column-wise defaults.
        # If the scaler contains `mean_` for each column (saved at training), use that as a sensible default
        # so missing inputs are imputed with the training mean rather than zeros.
        # Use a copy of scaler.mean_ so we don't accidentally mutate the scaler's internal
        # mean_ array when we change values in the input_series. Modifying scaler.mean_
        # directly will break subsequent scaling (it would make (x-mean)==0).
        if hasattr(scaler, 'mean_') and len(getattr(scaler, 'mean_')) == len(columns):
            input_series = pd.Series(getattr(scaler, 'mean_').copy(), index=columns, dtype=float)
        else:
            input_series = pd.Series(0, index=columns, dtype=float)

        # Helper to set categorical one-hot columns: set matched one to 1 and set other same-prefix columns to 0
        # This ensures a proper one-hot vector even when defaults (means) are present.
        def set_one_hot(key, val):
            key_prefix = f"{key}_"
            matched = False
            for col in columns:
                if col.startswith(key_prefix):
                    suffix = col[len(key_prefix):]
                    if str(suffix).strip().lower() == str(val).strip().lower():
                        input_series[col] = 1.0
                        matched = True
                    else:
                        # override any default/mean for other categories in this group
                        input_series[col] = 0.0
            return matched

        for key, val in data.items():
            # If the exact feature name exists (numeric feature), set it directly
            if key in input_series.index:
                try:
                    input_series[key] = float(val)
                except Exception:
                    # if it's not numeric, leave as-is (0) or try one-hot mapping
                    set_one_hot(key, val)
            else:
                # Try to map categorical to one-hot columns (e.g., Gender -> Gender_Male)
                set_one_hot(key, val)

        # Special handling for boolean-like fields commonly encoded as drop_first (e.g., Resigned_True)
        # If the API sends Resigned as True/False or "Yes"/"No", ensure the *_True column gets set
        if 'Resigned' in data and 'Resigned_True' in input_series.index:
            v = data.get('Resigned')
            if isinstance(v, bool):
                input_series['Resigned_True'] = 1.0 if v else 0.0
            else:
                if str(v).strip().lower() in ('yes', 'true', '1'):
                    input_series['Resigned_True'] = 1.0
                else:
                    input_series['Resigned_True'] = 0.0

        # Convert to single-row DataFrame (preserves feature names) and scale
        input_df = pd.DataFrame([input_series])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        return jsonify({
            "status": "success",
            "prediction": int(prediction)
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# ---------------------------------------
# Run Flask App
# ---------------------------------------
if __name__ == '__main__':
    print("🚀 Flask API starting...")
    app.run(host='127.0.0.1', port=5002, debug=True)
