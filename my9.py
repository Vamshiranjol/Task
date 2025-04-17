import numpy as np
import pandas as pd 
from flask import Flask, request, jsonify
import pickle

from sklearn.preprocessing import StandardScaler

with open('model.pkl','rb') as f:
    model= pickle.load(f)
    
with open('scaler.pkl','rb') as f:
    scaler= pickle.load(f)
    
app=Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        gender = data['gender']
        age = data['age']
        hypertension = data['hypertension']
        heart_disease = data['heart_disease']
        ever_married = data['ever_married']
        work_type = data['work_type']
        residence_type = data['Residence_type']
        avg_glucose_level = data['avg_glucose_level']
        bmi = data['bmi']
        smoking_status = data['smoking_status']
        
       
        features=np.array([[gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]])
        
        print(f"Features shape before scaling: {features.shape}")

        features_scaled = scaler.transform(features)
        
        prediction= model.predict(features_scaled)
        
        response={'prediction':int(prediction[0])}
        return jsonify(response)
    
    except Exception as e:
        response={'error':str(e)}
        return jsonify(response)
    
if __name__=='__main__':
    app.run(debug=True)
    