from flask import Flask,request,jsonify
import joblib
import pandas as pd
import numpy as np
app= Flask (__name__)
model= joblib.load('model.pkl')

@app.route('/')
def index():
    return "Welcome to ML Model Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data=request.get_json(force=True)

        age= data['age']
        salary=data['salary']

        prediction=model.predict(['age','salary'])
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)




