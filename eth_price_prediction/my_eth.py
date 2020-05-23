import json
from tensorflow import keras
import numpy as np
import tensorflow as tf
import time
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LayerNormalization
from Api.data_preprocessing import IncreaseScaler

with tf.name_scope('tool'):
    # 图层归一化
    class LNGRUCell(keras.layers.Layer):
        def __init__(self, units, activation="tanh", **kwargs):
            super().__init__(**kwargs)
            self.state_size = units
            self.output_size = units
            self.GRU_cell = keras.layers.GRUCell(units, activation=None)  # SimpleRNNCell
            self.layer_norm = LayerNormalization()
            self.activation = keras.activations.get(activation)

        def call(self, inputs, states):
            outputs, new_states = self.GRU_cell(inputs, states)
            norm_outputs = self.activation(self.layer_norm(outputs))
            return norm_outputs, [norm_outputs]

with tf.name_scope('data_processor'):
    with open("poloniex.json") as f:
        data = json.load(f)

    closing = [d["close"] for d in data]
    print(len(closing))

    # closing = np.array(closing[-10000:])
    closing = np.array(closing)

    input_step = 40
    output_step = 1
    series_step = input_step + output_step
    step = 3

    # 增幅标准化
    time_step = 40
    scaler = IncreaseScaler(time_step + 1, output_step)     # (41, 1)
    X, Y = scaler.train_normalize(closing, step)            # X (m, 40), Y (m, 1)
    print("max abs. X_train", np.max(np.abs(X)), np.min(np.abs(X)))
    print("max abs. Y_train", np.max(np.abs(Y)), np.min(np.abs(Y)))

    X = X[..., np.newaxis].astype(np.float32)   # X (m, 40, 1)
    Y = Y.reshape(Y.shape[0], )                 # y (m, )

    n_train = int(0.7 * len(X))
    n_valid = int(0.9 * len(X))

    X_train, X_valid, X_test = X[:n_train], X[n_train:n_valid], X[n_valid:]
    Y_train, Y_valid, Y_test = Y[:n_train], Y[n_train:n_valid], Y[n_valid:]

with tf.name_scope('train'):
    lr = 0.001
    n_cell = 40
    batch_size = 64
    dropout = 0.0
    loss_func = 'mae'

    model = keras.models.Sequential([
        keras.layers.GRU(n_cell, input_shape=(None, X_train.shape[2]), return_sequences=True,
                         # dropout=dropout, recurrent_dropout=dropout,
                         # kernel_regularizer=keras.regularizers.l2(0.01),
                         ),
        # keras.layers.BatchNormalization(),
        keras.layers.GRU(n_cell,
                         # dropout=dropout, recurrent_dropout=dropout,
                         # kernel_regularizer=keras.regularizers.l2(0.01),
                         ),
        # keras.layers.BatchNormalization(),
        # keras.layers.Dense(1, activation='tanh'),
        keras.layers.Dense(1),
    ])

    s = 20 * len(X_train) // batch_size  # number of steps in 20 epochs (batch size = 32)
    learning_rate = keras.optimizers.schedules.ExponentialDecay(lr, s, 0.1)
    optimizer = keras.optimizers.Adam(learning_rate, clipvalue=1.0)
    model.compile(loss=loss_func, optimizer=optimizer)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    start = time.time()
    history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=50,
                        batch_size=batch_size, callbacks=[early_stopping])
    end = time.time()
    print('Running time: %s Seconds' % (end - start))

with tf.name_scope('visualization'):
    forecast = model.predict(X_test[:100])
    actual = Y_test[:100]
    time = np.linspace(0, actual.shape[0], actual.shape[0])

    plt.plot(time, actual, label="Actual")
    plt.plot(time, forecast, label="Forecast")
    plt.legend(fontsize=14)
    plt.show()

    pd.DataFrame(history.history).plot()
    plt.show()

loss = np.mean(keras.losses.mean_squared_error(actual, forecast))
print('{0}: {1}, lr: {2}, n_cell: {3}, time_step: {4}, batch_size: {5}, dropout: {6}'
      .format(loss_func, loss, lr, n_cell, time_step, batch_size, dropout))
# mae: 0.0034301220439374447, lr: 0.001, n_cell: 40, time_step: 40, batch_size: 64, dropout: 0.0
# mae: 0.004396283999085426, lr: 0.001, n_cell: 40, time_step: 50, batch_size: 64, dropout: 0.0
