import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

# Load the trained model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON request
        data = request.get_json()

        # Validate input keys
        required_fields = [
            "gender", "age", "hypertension", "heart_disease",
            "ever_married", "work_type", "Residence_type",
            "avg_glucose_level", "bmi", "smoking_status"
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Extract and prepare input features
        features = np.array([[
            data["gender"],
            data["age"],
            data["hypertension"],
            data["heart_disease"],
            data["ever_married"],
            data["work_type"],
            data["Residence_type"],
            data["avg_glucose_level"],
            data["bmi"],
            data["smoking_status"]
        ]])

        # Log feature shape
        print(f"Features shape before scaling: {features.shape}")

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]

        # Return response
        response = {"prediction": int(prediction)}
        return jsonify(response), 200
    

    except Exception as e:
        # Log the error
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
