##Use decision trees to analyze patient data and predict disease categories.
# Visualize the decision tree and interpret the output. 
import numpy as np
import pandas as pd
import seaborn as sns
data=pd.read_csv('synthetic_patient_data.csv')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree
import matplotlib.pyplot as plt
np.random.seed(42)
num_samples = 200
data = {
    "Age": np.random.randint(20, 80, num_samples),
    "BMI": np.random.uniform(18, 35, num_samples),
    "Blood_Pressure": np.random.randint(90, 160, num_samples),
    "Cholesterol": np.random.randint(150, 300, num_samples),
    "Smoking_Habit": np.random.choice([0, 1], num_samples),  # 0: No, 1: Yes
    "Diabetes": np.random.choice([0, 1], num_samples),  # 0: No, 1: Yes
}
data["Disease"] = np.random.choice([0, 1, 2, 3], num_samples)
df=pd.DataFrame(data)
x=df.drop(columns=['Disease'])
y=df['Disease']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42) 
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(x_train, y_train)

plt.figure(figsize=(15, 8))
tree.plot_tree(clf, feature_names=x.columns, class_names=["No Disease", "Heart Disease", "Diabetes", "Hypertension"], filled=True, fontsize=10)
plt.show()

tree_rules = export_text(clf, feature_names=list(x.columns))
