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
savedtest = np.zeros((len(data), len(TARGETS)))
# List to store training and validation loss for each fold
train_loss_per_fold = []
val_loss_per_fold = []
if not os.path.exists('evaluation'):
    os.makedirs('evaluation')
if not os.path.exists('predictions'):
    os.makedirs('predictions')
if not os.path.exists('model_weights'):
    os.makedirs('model_weights')
# Cross-validation and Training
for fold_num, (train_index, test_index) in enumerate(tscv.split(X)):
    print(f"Fold {fold_num + 1}/{tscv.get_n_splits()}")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Build the model
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(window_size, len(FEATURES))))
    model.add(Dense(units=len(TARGETS)))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Callback to save model weights during training
    checkpoint = ModelCheckpoint(f'model_weights/best_model_weights_fold{fold_num}.h5',
                                 save_best_only=True, save_weights_only=True, verbose=1)

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32,
                        validation_data=(X_test, y_test), callbacks=[checkpoint])

    # Save training and validation loss
    train_loss_per_fold.append(history.history['loss'])
    val_loss_per_fold.append(history.history['val_loss'])

    # Load best model weights and make predictions
    model.load_weights(f'model_weights/best_model_weights_fold{fold_num}.h5')
    predictions = model.predict(X_test)
    # predictions = scaler.inverse_transform(predictions)
    # y_test = scaler.inverse_transform(y_test)
    savedtest[test_index] = predictions



    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print("RMSE:", rmse)
    print('---')

    # Plot training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f"Fold {fold_num + 1} - Training and Validation Loss")
    plt.legend()
    plt.savefig(f'evaluation/loss_curve_fold{fold_num}.png')
    plt.close()

    # Plot true and predicted values
    for i in range(len(TARGETS)):
        target_col_index = i
        y_pred = predictions[:, target_col_index]
        y_true = y_test[:, target_col_index]

        plt.figure(figsize=(8, 6))
        plt.plot(y_true, label='True')
        plt.plot(y_pred, label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(TARGETS[i])
        plt.legend()
        plt.savefig(f'predictions/predictions_fold{fold_num}_target{i}.png')
        plt.close()
print("Validation predictions:")
print(savedtest)
print("oof",savedtest.shape)
mn = 1.002049802661262
sd = 0.049110348928488375
# UNDO STANDARDIZATION
oofresult = savedtest.copy()
oofresult = (oofresult*sd)+mn
for i in range(5):
    print(oofresult[:,i].mean(),f' is average predicted future month {i+1} divided by future month {i}')
average_predictions = oofresult.mean(axis=0)
months = np.arange(1, 6)  # Assuming you have predictions for 5 future months
plt.plot(months, average_predictions)
plt.xlabel('Future Month')
plt.ylabel('Average Predicted Value')
plt.title('Average Predicted Future Month Value')
plt.grid(True)
plt.show()