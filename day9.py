import numpy as np
import pandas as pd
import seaborn as sns
data=pd.read_csv('synthetic_patient_data.csv')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
np.random.seed(42)
num_samples = 200
data = {
    "Age": np.random.randint(20, 80, num_samples),
    "BMI": np.random.uniform(18, 35, num_samples),
    "Blood_Pressure": np.random.randint(90, 160, num_samples),
    "Cholesterol": np.random.randint(150, 300, num_samples),
    "Smoking_Habit": np.random.choice([0, 1], num_samples),  # 0: No, 1: Yes
    "Diabetes": np.random.choice([0, 1], num_samples),  # 0: No, 1: Yes
}
data["Disease"] = np.random.choice([0, 1, 2, 3], num_samples)
df=pd.DataFrame(data)
x=df.drop(columns=['Disease'])
y=df['Disease']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42) 
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(x_train, y_train)

rf_clf= RandomForestClassifier(n_estimators=100,max_depth=4,random_state=42)
rf_clf.fit(x_train, y_train)
 ##predictions
dt_preds=clf.predict(x_test)
rf_preds=rf_clf.predict(x_test)
## accuracy
dt_accuracy=accuracy_score(y_test,dt_preds)
rf_accuracy=accuracy_score(y_test, rf_preds)
## classification reports
dt_report=classification_report(y_test, dt_preds, target_names=(["No Disease", "Heart Disease", "Diabetes", "Hypertension"]))
rf_report=classification_report(y_test, rf_preds, target_names=(["No Disease", "Heart Disease", "Diabetes", "Hypertension"]))
print(dt_accuracy,rf_accuracy)
print(dt_report,rf_report)