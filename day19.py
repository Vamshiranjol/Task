#Perform feature importance analysis to identify key factors influencing diagnosis.
#Visualize important features using SHAP or LIME.

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df=pd.read_csv('diabetes.csv')
X=df.drop('Outcome',axis=1)
y=df['Outcome']
X_train_raw,X_test_raw,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train_raw)
X_test=scaler.transform(X_test_raw)

model=RandomForestClassifier(n_estimators=200,random_state=42)
model.fit(X_train,y_train)

explainer=shap.Explainer(model)
shap_values=explainer(X_test_raw)

X_test_df = pd.DataFrame(X_test, columns=X.columns)

# shap.summary_plot(shap_values[1], X_test_df, plot_type="dot", show=True)

shap.plots.bar(shap_values[:, :, 1])
shap.plots.beeswarm(shap_values[:, :, 1])









