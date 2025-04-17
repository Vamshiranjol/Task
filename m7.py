import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
pio.templates.default ="plotly_white"
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf     


data=pd.read_csv('Instagram-Reach.csv')
print(data.head())
print(data.tail())

print(data.info())
print(data.isna().sum())
data["Date"]=pd.to_datetime(data["Date"])
print(data["Date"])
data["Date1"]=data["Date"].dt.date
print(data["Date1"])

fig=go.Figure()
fig.add_trace(go.Scatter(x=data['Date'],
                          y=data['Instagram reach'],
                          mode='lines', name='Instagram reach'))
fig.update_layout(title="vamshi", xaxis_title="Date", yaxis_title="Instagram reach")
fig.show()

time_series=data.set_index("Date")["Instagram reach"]
print(time_series)
differenced_series=time_series.diff().dropna()
print(differenced_series)




fig, axes=plt.subplots(1,2, figsize=(12,4))
plot_acf(differenced_series, ax=axes[0])
plot_pacf(differenced_series, ax=axes[1])
plt.show()





