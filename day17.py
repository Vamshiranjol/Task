# Build a Recurrent Neural Network (RNN) for predicting patient vitals over time.
# Train the RNN and visualize predictions.    
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt 
import streamlit as st
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import SimpleRNN, Dense 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv('sample_patient_vitals.csv')

heart_rate = df['HeartRate'].values.reshape(-1, 1)
scaler = MinMaxScaler()

scaled_heart_rate = scaler.fit_transform(heart_rate)

def create_sequences(data, seq_length=5):
    X, y=[], []
    for i in range(len(data)- seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X),np.array(y)
seq_length = 5
X, y = create_sequences(scaled_heart_rate, seq_length)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


model=Sequential([
    SimpleRNN(32,input_shape=(seq_length, 1), activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=50, batch_size=8,
                    validation_data=(X_test, y_test), verbose=0)

   
st.set_page_config(page_title="Heart Rate Predictor")
st.title("Predict Next Heart Rate using RNN")

st.write("Enter the last 5 heart rate readings:")


input_hr = []
for i in range(seq_length):
    value = st.number_input(f"Heart Rate #{i+1}", min_value=30.0, max_value=200.0, value=70.0)
    input_hr.append(value)


if st.button("Predict Next Heart Rate"):
    input_array = np.array(input_hr).reshape(-1, 1)
    input_scaled = scaler.transform(input_array)
    input_seq = input_scaled.reshape(1, seq_length, 1)

    predicted_scaled = model.predict(input_seq)
    predicted_hr = scaler.inverse_transform(predicted_scaled)

    st.success(f" Predicted Next Heart Rate: **{predicted_hr[0][0]:.2f} bpm**")


# predicted = model.predict(X_test)


# predicted_hr = scaler.inverse_transform(predicted)
  
# actual_hr = scaler.inverse_transform(y_test.reshape(-1,1))



# fig, ax = plt.subplots(figsize=(10, 5))
# ax.plot(actual_hr, label='Actual Heart Rate')
# ax.plot(predicted_hr, label='Predicted Heart Rate')
# ax.set_title("Heart Rate Prediction using RNN")
# ax.set_xlabel("Time Steps")
# ax.set_ylabel("Heart Rate")
# ax.legend()
# ax.grid(True)

# st.pyplot(fig)

