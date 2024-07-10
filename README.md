# Nifty50-Prediction

Nifty Stock Price Prediction:
This project demonstrates a method for predicting the high and low prices of Nifty stocks using a Stacked LSTM neural network. The model is trained on historical stock data to predict future prices. 

Table of Contents
Introduction
Data
Preprocessing
Model
Training
Evaluation
Usage
Requirements

Introduction
This project aims to predict the high and low prices of Nifty stocks using Stacked LSTM neural network. The LSTM model is chosen for its ability to handle time series data and capture temporal dependencies.

Data
The dataset is used in this project contains historical Nifty stock prices which is taken from the NSE official website. It includes various columns such as High, Low, Open, Close, and Volume.

Preprocessing
Loading Data: The data is loaded using pandas.
Checking for Missing Values: Missing values and duplicate rows are checked and handled.
Scaling Data: The High and Low columns are normalized using MinMaxScaler to bring values within the range of 0 to 1.
python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

# Scale High values
high_values = ds0["High"].values.reshape(-1, 1)
ds0["High"] = scaler.fit_transform(high_values)

# Scale Low values
low_values = ds0["Low"].values.reshape(-1, 1)
ds0["Low"] = scaler.fit_transform(low_values)

Model
An LSTM model is constructed using Keras. The model consists of three LSTM layers with dropout to prevent overfitting and two dense layers for the output.

python
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(100, 1)))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

Training
The data is split into training and testing sets with a 70-30 ratio. The model is trained for 100 epochs with a batch size of 24.

python
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=24, verbose=1)

Evaluation
The model's performance is evaluated using Mean Squared Error (MSE). The predictions are plotted against the actual values to visualize the accuracy.

python
from sklearn.metrics import mean_squared_error
import math

train_predict = scaler.inverse_transform(model.predict(X_train))
test_predict = scaler.inverse_transform(model.predict(X_test))

train_score = math.sqrt(mean_squared_error(Y_train, train_predict))
test_score = math.sqrt(mean_squared_error(Y_test, test_predict))

print(f'Train Score: {train_score} RMSE')
print(f'Test Score: {test_score} RMSE')

Usage
To use this project, follow these steps:

Clone the repository.
Install the required dependencies.
Run the Jupyter notebook or script.

Requirements
Python 3.x
pandas
numpy
matplotlib
scikit-learn
keras
tensorflow
Install the required packages using:

bash
Copy code
pip install pandas numpy matplotlib scikit-learn keras tensorflow
