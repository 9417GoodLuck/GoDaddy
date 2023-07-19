import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


data = pd.read_csv('train_data.csv')

FEATURES = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']
TARGETS = ['y0', 'y1', 'y2', 'y3', 'y4']
features = data[FEATURES].values
targets = data[TARGETS].values


scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
scaled_targets = scaler.fit_transform(targets)


window_size = 10


X = []
y = []
for i in range(window_size, len(scaled_features)):
    X.append(scaled_features[i-window_size:i])
    y.append(scaled_targets[i])

X = np.array(X)
y = np.array(y)


tscv = TimeSeriesSplit(n_splits=5)


model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(window_size, len(FEATURES))))
model.add(Dense(units=len(TARGETS)))


model.compile(optimizer='adam', loss='mean_squared_error')

# makedir to save model
if not os.path.exists('model_weights'):
    os.makedirs('model_weights')

# the callback to save model
checkpoint = ModelCheckpoint('model_weights/best_model_weights.h5', save_best_only=True, save_weights_only=True, verbose=1)

# cross vaild
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


    model.fit(X_train, y_train, epochs=10, batch_size=32, callbacks=[checkpoint])

    model.save_weights('model_weights/best_model_weights.h5')

    model.load_weights('model_weights/best_model_weights.h5')

    predictions = model.predict(X_test)

    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)


    # evaluation
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print("RMSE:", rmse)
    print('---')

    # plot
    for i in range(len(TARGETS)):
        target_col_index = i
        y_pred = predictions[:, target_col_index]
        y_true = y[:, target_col_index]

        plt.plot(y_true, label='True')
        plt.plot(y_pred, label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(TARGETS[i])
        plt.legend()
        plt.show()
