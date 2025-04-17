import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
data= {
    "age" : [25,35,37,40,45,47],
    "salary": [27000,35000,38000,40000,42000,45000],
    "purchased":[0,0,1,1,1,0]
}

df=pd.DataFrame(data)

x=df[['age','salary']]
y=df['purchased']

model=RandomForestClassifier()
model.fit(x,y)

joblib.dump(model, 'model.pkl')


