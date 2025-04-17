import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("medicalappointmentsnoshow.csv")
print(data.head())
print(data.isna().sum())
print(data.info())

data['PatientId'] = data['PatientId'].astype('int')
print(data['PatientId'])


data['Gender']=data['Gender'].map({'F': 1, 'M': 0})
print(data['Gender'])
data['Gender'] = data['Gender'].astype(int)
data['No-show']=data['No-show'].map({'Yes': 1, 'No': 0})
print(data['No-show'])
data['No-show'] = data['No-show'].astype(int)


data['SMS_received'] = data['SMS_received'].astype('int64')
print(data['SMS_received'])

data['ScheduledDay'] = data['ScheduledDay'].astype('category')
print(data['ScheduledDay'])

data['ScheduledDay'] = pd.to_datetime(data['ScheduledDay'])

# Step 2: Extract useful date-time features
data['Year'] = data['ScheduledDay'].dt.year
data['Month'] = data['ScheduledDay'].dt.month
data['Day'] = data['ScheduledDay'].dt.day
data['Hour'] = data['ScheduledDay'].dt.hour
data['Minute'] = data['ScheduledDay'].dt.minute
data['Second'] = data['ScheduledDay'].dt.second

# Step 3: Optionally, drop the original 'Timestamp' column (if you don't need it anymore)
data = data.drop(columns=['ScheduledDay'])

# Display the modified dataframe with the extracted features
print(data)
data['AppointmentDay'] = pd.to_datetime(data['AppointmentDay'])

# Step 2: Extract useful date-time features
data['Year'] = data['AppointmentDay'].dt.year
data['Month'] = data['AppointmentDay'].dt.month
data['Day'] = data['AppointmentDay'].dt.day
# data['Hour'] = data['AppointmentDay'].dt.hour
# data['Minute'] = data['AppointmentDay'].dt.minute
# data['Second'] = data['AppointmentDay'].dt.second

# Step 3: Optionally, drop the original 'Timestamp' column (if you don't need it anymore)
data1 = data.drop(columns=['AppointmentDay'])
print(data1)

print(data['Neighbourhood'].unique())

X = data.drop(['No-show', 'PatientId', 'AppointmentID'], axis=1)
y = data['No-show']
print("Features (X):")
print(X.head())

print("\n target (y):")
print(y.head())

print(data.info())

from sklearn.model_selection import train_test_split

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
