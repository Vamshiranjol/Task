import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('dataset.csv')
print(df.head())
print(df.isna().sum())

X=df.drop('target', axis=1)
y=df['target']

print("Features (X):")
print(X.head()) 

print("\nTarget (y):")
print(y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)
print(classification_report(y_test, y_pred))

