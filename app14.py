import numpy as np
import pandas as pd
from flask import Flask,request,render_template
import pickle

flask_app=Flask(__name__)


with open('day14.pkl', 'rb') as f:
    model = pickle.load(f)


@flask_app.route('/')
def Home():
    return render_template('index14.html')
@flask_app.route('/predict',methods=['POST'])
def predict():
    float_feature=[float(x) for x in request.form.values()]


    final_input = [float_feature]
    Prediction=model.predict(final_input)[0]

    result = {
        0: "High Risk",
        1: "Medium Risk",
        2: "Low Risk",
        3: "Very Low Risk"
    }

    return render_template('index14.html',prediction_text=f'Prediction: {result[Prediction]}')
if __name__ =="__main__":
     flask_app.run(debug=True)