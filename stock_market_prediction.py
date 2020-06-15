import math

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from keras import backend as K
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def create_dataset(data, time_step=1):
    X_train = []
    Y_train = []
    for i in range(len(data) - time_step - 1):
        X_train.append(data[i:(i + time_step), 0])
        Y_train.append(data[i + time_step, 0])

    return np.array(X_train), np.array(Y_train)


# download data from Yahoo Finance API
stock = yf.Ticker("AMZN") # netflix
# get last 5 years of data
stock_data = stock.history(period="3Y")

# dropping unwanted columns
# stock_data.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
# getting only useful cols in data
data = stock_data['Close']

# plotting the data along time
plt.plot(data)
plt.show()

# scale data between 0 to 1, so as to make it on the same scale.
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(np.array(data).reshape(-1, 1))  # data will be in form of ndarray now

# defining train and test sizes for splitting of data
train_size = int(len(data) * 0.65)
test_size = len(data) - train_size

# split data for training and testing
train_data = data[:train_size, :]
test_data = data[train_size:len(data), :]

time_step = 150  # these many days of data will be considered to predict the next day's data. The more the better
# training data
X_train, Y_train = create_dataset(train_data, time_step)  # 2d array
# testing data
X_test, Y_test = create_dataset(test_data, time_step)  # 2d array

# convert train and test data in 3D, as an input to LSTM model
# reshaping data into (samples, time steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],
                          1)  # adding '1' as the 3rd dimension, 1st and 2nd are samples and time steps
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],
                        1)  # adding '1' as the 3rd dimension, 1st and 2nd are samples and time steps

# clearing the keras session before building model
K.clear_session()
# creating LSTM model
model = Sequential()
# Stacked LSTM, means one LSTM on top of another LSTM
# adding layers
model.add(LSTM(50, return_sequences=True, input_shape=(
    X_train.shape[1], X_train.shape[2])))  # hidden layer = 50, input shape = time_steps and last dimension in X_train
# second LSTM
model.add(LSTM(50, return_sequences=True))
# third LSTM
model.add(LSTM(50))
# adding output layer
model.add(Dense(1))
# compiling the whole model
model.compile(loss='mean_squared_error', optimizer='adam')
# checking model summary
print(model.summary())

# fit the model using parameters
epochs = 150
batch_size = 64
verbose = 1
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size, verbose=verbose)

# predicting on test data and checking performance metrics
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# transforming data back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# calculating performance metrics on training data
print(math.sqrt(mean_squared_error(Y_train, train_predict)))
# calculating performance metrics on testing data
print(math.sqrt(mean_squared_error(Y_test, test_predict)))

# plot the prediction with original data

# shift train predictions for plotting
trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_step:len(train_predict) + time_step, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict) + (time_step * 2) + 1:len(data) - 1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(data))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

x_input = test_data[len(test_data) - 100:].reshape(1, -1)  # 1 row, 100 features
temp_input = x_input.tolist()[0]  # convert ndarray to list format

# calculate prediction for next 30 days
days = 30
list_output = []
i = 0
while (i < days):
    if (len(temp_input) > 100):
        # print(temp_input)
        x_input = np.array(temp_input[1:])
        # print("{} day input {}".format(i, x_input))
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, time_step, 1))
        # print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i, yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        # print(temp_input)
        list_output.extend(yhat.tolist())
        i = i + 1
    else:
        x_input = x_input.reshape((1, time_step, 1)) # todo error
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        # print(len(temp_input))
        list_output.extend(yhat.tolist())
        i = i + 1

# plot the new days prediction
day_new=np.arange(1,time_step+1)
day_pred=np.arange(time_step+1,time_step+1+days)

plt.plot(day_new,scaler.inverse_transform(data[len(data)-time_step:]))
plt.plot(day_pred,scaler.inverse_transform(list_output))
plt.show()

# for continuos plot
data_to_plot=data.tolist()
data_to_plot.extend(list_output)
plt.plot(data_to_plot[1200:])
plt.show()

# transforming data_to_plot back to original scale
data_to_plot = scaler.inverse_transform(data_to_plot).tolist()
plt.plot(data_to_plot)
plt.show()