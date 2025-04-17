from flask import Flask,request,jsonify,render_template
import pickle

with open('disease_tablet.pkl','rb') as f:
    model=pickle.load(f)

app = Flask(__name__)
@app.route('/')
def Home():
    return render_template('index13.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptom = request.form['symptom'].strip().lower()
    

    prediction = model.predict([symptom])[0]
    return render_template('index13.html', prediction_text=f"Suggested Tablet: {prediction}")
    
    

if __name__ == "__main__":
    app.run(debug=True)

