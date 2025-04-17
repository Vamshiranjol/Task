import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

test=pd.read_csv('healthcare-dataset-stroke-data.csv')
print(test)
print(test.head())
print(test.isna().sum())

test1=test['bmi'].unique()
print(test1)
print(test.info())



# sns.histplot(test['smoking_status'], kde=True)
# plt.title('Histogram and KDE Plot')
# plt.show()

# # Q-Q Plot
# import scipy.stats as stats
# stats.probplot(test['smoking_status'], dist="norm", plot=plt)
# plt.show()

# Calculate skewness
# skewness = test['bmi'].skew()
# print(f"Skewness: {skewness}")

# # Interpret skewness
# if skewness > 0.5:
#     print("Data is positively skewed.")
# elif skewness < -0.5:
#     print("Data is negatively skewed.")
# else:
#     print("Data is approximately symmetric.")
    
    
test['smoking_status']=test['smoking_status'].map({'formerly smoked': 1, 'never smoked': 2, 'smokes': 3, 'Unknown': 0})
print(test['smoking_status'])
test['smoking_status'] = test['smoking_status'].astype(int)

test2=test['ever_married'].unique()
print(test2)

test['ever_married']=test['ever_married'].map({'Yes': 1, 'No': 0})
print(test['ever_married'])
test['ever_married']=test['ever_married'].astype(int)


test['Residence_type']=test['Residence_type'].map({'Urban': 1, 'Rural': 0})
print(test['Residence_type'])
test['Residence_type']=test['Residence_type'].astype(int)


test['work_type']=test['work_type'].map({'Never_worked': 0, 'Self-employed': 1, 'children': 3, 'Govt_job': 4, 'Private': 5})
print(test['work_type'])
test['work_type']=test['work_type'].astype(int)
test4=test['gender'].unique()
print(test4)


test['gender']=test['gender'].map({'Male': 1, 'Female': 0, 'Other': 5})
print(test['gender'])
test['gender']=test['gender'].astype(int)
print(test.info())


# Fill missing values with the mean
test['bmi'] = test['bmi'].fillna(test['bmi'].median())
print(test['bmi'])

test5=test.drop(columns=['id'], inplace=True)

print(test5)

print(test.info())
row = test.iloc[5]
print(row)

# Example: model prediction returns a probability
probability = 0.6  # This would be returned by the API
if probability >= 0.5:
    print("Prediction: Stroke (1)")
else:
    print("Prediction: No Stroke (0)")


X = test.drop(['stroke'], axis=1)
y = test['stroke']

print("Features (X):")
print(X.head()) 


print("\nTarget (y):")
print(y.head())

from sklearn.base import accuracy_score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=47)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled)
print(X_test_scaled )

model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
X_train_predicition=model.predict(X_train)
training_accuracy=accuracy_score(X_train_predicition, y_train)
print(f"The training accuracy is {training_accuracy}")


# accuracy = model.score(X_test_scaled, y_test)
# print(f"Accuracy: {accuracy * 100:.2f}%")
 

# with open('model.pkl','wb') as f:
#      pickle.dump(model,f)
# print("Model saved to the 'model.pkl'")  

# with open('scaler.pkl', 'wb') as f:
#     pickle.dump(scaler, f)
       
    
