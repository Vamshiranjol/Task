import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('diabetes.csv')
x=df.drop("Outcome",axis=1)
y=df['Outcome']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)



models={"logistic_regression": LogisticRegression(max_iter=1000),
        "support_vector_machine":SVC(),
        "neural_network":MLPClassifier(max_iter=1000,random_state=42),
        "random_forest":RandomForestClassifier(n_estimators=100,random_state=42)}

results=[]

for name,model in models.items():
    model.fit(x_train_scaled,y_train)
    y_predict=model.predict(x_test_scaled)
    acc=accuracy_score(y_test,y_predict)
    results.append({'models':name,'Accuracy':round(acc*100,2)})

report_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

print("\nModel Comparison Report:\n")
print(report_df)





