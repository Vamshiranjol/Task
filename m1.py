import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv('healthmonitoring.csv')
print(data)
print(data.head())
print(data.tail())
print(data.info())
# print(data.isna().sum())

re1=data['BodyTemperature'].unique()
print(re1)
# Fill missing values with the median
# Fill missing values with the mean of the column
data['BodyTemperature'] = data['BodyTemperature'].fillna(data['BodyTemperature'].median())

# Alternatively, you could use the median or mode
# data['bodytemperature'] = data['bodytemperature'].fillna(data['bodytemperature'].median())
# data['bodytemperature'] = data['bodytemperature'].fillna(data['bodytemperature'].mode()[0])

# Print the result
print(data['BodyTemperature'])
re=data['OxygenSaturation'].unique()
print(re)
data['OxygenSaturation'] = data['OxygenSaturation'].fillna(data['OxygenSaturation'].median())
print(data['OxygenSaturation'])

print(data.isna().sum())


plt.figure(figsize=(8, 6))
sns.barplot(data=data, x='Age', y='Gender', estimator='mean')
plt.title('Age Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()

plt.figure(figsize=(8, 8))
sns.lineplot(data=data, x='Age', y='HeartRate', estimator='mean')
plt.title('heartrate by age')
plt.xlabel('Age')
plt.ylabel('HeartRate')
plt.show()






