import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier
data=pd.read_csv('patient_data.csv')
print(data)
print (data.head())
x=data.drop(['Disease'], axis= 1)
y=data['Disease']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x)
print(y)

model=RandomForestClassifier()
model.fit(x_train,y_train)
predictions=model.predict(x_test)
accuracy=model.score(x_test,y_test)
print('Accuracy:',accuracy)


# Save the model to a file
with open('day14.pkl', 'wb') as f:
    pickle.dump(model, f)

