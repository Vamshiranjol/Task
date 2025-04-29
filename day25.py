import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Generate synthetic daily admissions data for 60 days
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", periods=60)
admissions = 50 + np.cumsum(np.random.normal(loc=1, scale=2, size=60)).astype(int)
data = pd.DataFrame({'date': dates, 'admissions': admissions})
data.set_index('date', inplace=True)

# Step 2: Fit ARIMA model (adjust order as needed)
model = ARIMA(data['admissions'], order=(2, 1, 2))
results = model.fit()

# Step 3: Forecast next 14 days
forecast = results.get_forecast(steps=14)
forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=14)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# Step 4: Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['admissions'], label='Observed')
plt.plot(forecast_index, forecast_mean, label='Forecast', color='orange')
plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.2)
plt.title('Synthetic Hospital Admissions Forecast (ARIMA)')
plt.xlabel('Date')
plt.ylabel('Admissions')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 5: Print forecasted values
forecast_df = pd.DataFrame({
    'date': forecast_index,
    'forecasted_admissions': forecast_mean.round().astype(int)
})
print(forecast_df)
