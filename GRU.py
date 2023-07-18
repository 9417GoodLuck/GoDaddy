# use tensorflow as the model frame
import tensorflow as tf
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import GroupKFold

VER = 98

# SET NONE TO TRAIN NEW MODEL
INFER_FROM_PATH = None

def build_model():
    # input layer
    inp = tf.keras.Input(shape=(WIDTH - COPIES - 4 - 1, 1))  # INPUT SHAPE IS 12

    # three GRU layers
    x = tf.keras.layers.GRU(units=8, return_sequences=True)(inp)
    x = tf.keras.layers.GRU(units=8, return_sequences=True)(x)
    x = tf.keras.layers.GRU(units=8, return_sequences=False)(x)
    # output layer, add a fully connected layer with 5 neurons and use linear activation
    x = tf.keras.layers.Dense(5, activation='linear')(x)  # OUTPUT SHAPE IS 5

    # The input layer and output layer are defined as the input and output of the model
    model = tf.keras.Model(inputs=inp, outputs=x)

    # Configure the loss function and optimizer into the model.
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(loss=loss, optimizer=opt)

    return model


# Define parameters
train_data = pd.read_csv('data/train_data.csv')
FEATURES = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']
TARGETS = ['y0', 'y1', 'y2', 'y3', 'y4']
WIDTH = 35
COPIES = 18

FOLDS = 20
VERBOSE = 2
oof = np.zeros((len(train_data), 5))
if not os.path.exists('models'):
    os.mkdir('models')

# Create GroupKFold object
skf = GroupKFold(n_splits=FOLDS)
skf = skf.split(train_data, train_data['y0'], train_data['cfips'])

# Loop over each fold
for fold, (train_idx, valid_idx) in enumerate(skf):
    print('#' * 25)
    print('### Fold', fold + 1)
    print('### Train size', len(train_idx), 'Valid size', len(valid_idx))
    print('#' * 25)

    # Train, validation data split
    X_train = train_data.loc[train_idx, FEATURES]
    y_train = train_data.loc[train_idx, TARGETS]
    X_valid = train_data.loc[valid_idx, FEATURES]
    y_valid = train_data.loc[valid_idx, TARGETS]

    X_train = np.expand_dims(X_train, axis=2)
    y_train = np.expand_dims(y_train, axis=2)
    X_valid = np.expand_dims(X_valid, axis=2)
    y_valid = np.expand_dims(y_valid, axis=2)

    # Weight recent sample more
    GRP = len(train_idx) // COPIES
    w = np.array([1] * (COPIES - 7) + [1, 1, 2, 2] + [2, 2, 2])
    w = w / np.sum(w)

    # Build the model
    model = build_model()

    # Train the model
    if INFER_FROM_PATH is None:
        h = model.fit(X_train, y_train,
                      validation_data=(X_valid, y_valid),
                      sample_weight=np.tile(w, GRP),
                      batch_size=4, epochs=2, verbose=VERBOSE)
    else:
        model.load_weights(INFER_FROM_PATH + f'GRU_f{fold}_v{VER}.h5')

    # Save model weights
    model.save_weights(f'models/GRU_f{fold}_v{VER}.h5')

    # Predict and fill oof
    oof[valid_idx, :] = model.predict(X_valid, verbose=VERBOSE)

