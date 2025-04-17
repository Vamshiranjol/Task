import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

data=pd.read_csv("sample_submission.csv")
print(data)

print(data.isna().sum())
print(data.info())

X=data.drop('readmitted', axis=1)
y=data['readmitted']

print("Features (X) :")
print(X.head())

print("\n target (y):")
print(y.head())

X1=pd.get_dummies(X)
print(X1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier()
model.fit(X_train,y_train)
pred=model.predict(X_test)
print(pred)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, pred)

print(f'Accuracy: {accuracy}')

sns.countplot(x=pred, palette='Set2')

# Adding title and labels
plt.title('Number of Readmitted Patients (1: Readmitted, 0: Not Readmitted)')
plt.xlabel('Readmission Status')
plt.ylabel('Count')


# Display the plot
plt.show()

import plotly.graph_objects as go


# Count the number of readmitted and not readmitted patients
readmitted_count = np.count_nonzero(pred == 1)
not_readmitted_count = np.count_nonzero(pred == 0)

# Create a hierarchical tree-like structure for visualization
labels = ["Readmitted", "Not Readmitted"]
values = [readmitted_count, not_readmitted_count]

fig = go.Figure(go.Sunburst(
    labels=labels,
    parents=["", ""],  # "" means top-level parents
    values=values,
    marker=dict(colors=[0, 1], colorscale='Blues')
))

fig.update_layout(title="Readmission Status Tree", margin=dict(t=0, l=0, r=0, b=0))
fig.show()

