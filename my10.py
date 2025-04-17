import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

pf= pd.read_csv('healthcare-dataset-stroke-data.csv')
print(pf.head())
print(pf.isna().sum())
pf.drop(columns=['id'], inplace= True )
pf['bmi'] = pf['bmi'].fillna(pf['bmi'].mean())
print(pf['bmi'])
print(pf)
pf['smoking_status']=pf['smoking_status'].map({'formerly smoked': 1, 'never smoked': 2, 'smokes': 3, 'Unknown': 0})
print(pf['smoking_status'])
pf['smoking_status'] = pf['smoking_status'].astype(int)

test2=pf['ever_married'].unique()
print(test2)

pf['ever_married']=pf['ever_married'].map({'Yes': 1, 'No': 0})
print(pf['ever_married'])
pf['ever_married']=pf['ever_married'].astype(int)


pf['Residence_type']=pf['Residence_type'].map({'Urban': 1, 'Rural': 0})
print(pf['Residence_type'])
pf['Residence_type']=pf['Residence_type'].astype(int)


pf['work_type']=pf['work_type'].map({'Never_worked': 0, 'Self-employed': 1, 'children': 2, 'Govt_job': 3, 'Private': 4})
print(pf['work_type'])
pf['work_type']=pf['work_type'].astype(int)
test4=pf['gender'].unique()
print(test4)

pf= pf[pf['gender']!= 'Other']
pf['gender']=pf['gender'].map({'Male': 1, 'Female': 0}).astype(int)
print(pf['gender'])
print(pf.info())


X = pf.drop(columns=['stroke'])
y = pf['stroke']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X[['age', 'avg_glucose_level', 'bmi']] = scaler.fit_transform(X[['age', 'avg_glucose_level', 'bmi']])


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=47)

model = DecisionTreeClassifier(criterion = 'gini', max_depth = 5, min_samples_split=20, min_samples_leaf=10, random_state = 42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


tree_rules= export_text(model, feature_names=list(X.columns))
print(tree_rules)

import matplotlib.pyplot as plt
plt.figure(figsize=(15, 10))

plot_tree(model, feature_names = X.columns, class_names= ['No stroke', 'stroke'], filled = True, rounded = True, fontsize=8 )
plt.show()
