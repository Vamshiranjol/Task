import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data=pd.read_csv('titanic.csv')
print(data)

print(data.info())

re1=data['Age'].unique()
print(re1)
re2=data['Sex'].unique()
print(re2)
rey=data['Age'].fillna(data['Age'].median(), inplace=True)
print(rey)

re3=data.drop(['Cabin', 'Ticket'],axis=1, inplace=True)

re4=data['Embarked'].unique()

data['Embarked'] = data['Embarked'].replace({
    'S': '1',
    'C': '2',
    'Q': '3'
})

# Fill missing values in the 'Embarked' column with the most frequent value (mode)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Sex']=data['Sex'].map({'male':0, 'female':1})
print(data['Sex'].unique()) 


print(data.isna().sum())
print(data.head())
X = data.drop(['Survived'], axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
