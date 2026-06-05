'''
Stock Price Prediction Using Maching Learning Tutorial

From https://www.simplilearn.com/tutorials/machine-learning-tutorial/stock-price-prediction-using-machine-learning

## comments are section headers that can be searched for in the document
# Will be notes on function

This is the clean base version

Strategy:
1: Analyze predicted price as it comes in to determine if this is actually capable of a somewhat accurate prediction in real time.
        Big concern is that it is not a real time prediction, it's using it's own data to recreate something that already happened - which would be useless
2: Alter the algorithim here to highlight periods that it would want to buy
        Probably a positive first derivative of a linear fit of the predicted price
3: Refine buying algorithm to achieve TBD profit margin/risk profile
4: Define a panic sell function to protect profits- the end of the data set looks like a prime canidate        
5: Convert algo from tutorial dataset to static bitcoin dataset at tbd sample rate
6: Convert static bitcoin dataset to TBD number of previous samples with periodic retraining
        Idea is that the algos driving the price likely change strategy periodically
'''


## Import the Libraries

import pandas as pd
import os
import matplotlib.pyplot as plt
%matplotlib inline

## Load the Training Dataset

dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
dataset_train.head()

## Use the Open Stock Price Column to Train Your Model

training_set = dataset_train.iloc[:,1:2].values

print(training_set)
print(training_set.all)

## Normalizing the Dataset
      
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1))
scaled_training_set = scaler.fit_transform(training_set)

scaled_training_set

## Creating X_train and Y_train Data Structures

X_train = []
y_train = []
for i in range (60,1258):
    X_train.append(scaled_training_set[i-60:i, 0])
    y_train.append(scaled_training_set[i,0])
X_train = np.array(X_train)
y_train = np.array(y_train)

print(X_train.shape)
print(y_train.shape)

## Reshape the Data

X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1))
X_train.shape

## Building the Model by Importing the Crucial Libraries and Adding Different Layers to LSTM.

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences= True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences= True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences= True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))


##Fitting the Model

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

## Extracting the Actual Stock Prices of Jan-2017

dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
actual_stock_price = dataset_test.iloc[:,1:2].values

## Preparing the Input for the Model

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total)- len(dataset_test)-60:].values

inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

X_test = []
for i in range (60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 1))

## Predicting the Values for Jan 2017 Stock Prices

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)


## Plotting actual and Predicted Prices for Google Stocks

plt.plot(actual_stock_price, color = 'red', label = 'Actual Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend






