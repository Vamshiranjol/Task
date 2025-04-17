import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


data=pd.read_csv('Social_Network_Ads.csv')
print(data)
data.shape
columns=['User ID','Gender']
data=data.drop(columns=columns)
x=data.drop(['Purchased'],axis=1)
y=data['Purchased']
print(x)
print(y)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.25,random_state=42) 

##from sklearn.preprocessing import StandardScaler
#ss=StandardScaler()
#X_train=ss.fit_transform(X_train)
# X_test=ss.transform(X_test)

from sklearn.svm import SVC
classifier=SVC()
classifier.fit(X_train,Y_train)

Y_predict=classifier.predict(X_test)
print(accuracy_score(Y_predict,Y_test))



