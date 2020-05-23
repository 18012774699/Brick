import json
from tensorflow import keras
import numpy as np
import time
import matplotlib.pyplot as plt


with open("poloniex.json") as f:
    data = json.load(f)

closing = [d["close"] for d in data]
print(len(closing))

# 增幅标准化
zipped = zip(closing[0:-1], closing[1:])
changes = [d1 / d0 - 1 for d0, d1 in zipped]
closing = np.array(closing)
changes = np.array(changes)

length = 40
step = 3
sequences = np.empty((0, 40))
results = np.empty((0, 1))
for i in range(0, changes.shape[0] - length - 1, step):
    sequences = np.append(sequences, changes[i: i + length].reshape(-1, 40), axis=0)  # 2~41天增幅
    results = np.append(results, (closing[i + length + 1] / closing[i] - 1).reshape(-1, 1), axis=0)  # 第42相对第一天的增幅

print("max abs. sequences", np.max(np.abs(sequences)))
print("max abs. results", np.max(np.abs(results)))

sequences = sequences[..., np.newaxis].astype(np.float32)
n_train = int(0.7 * len(sequences))
n_test = int(0.1 * len(sequences))

X_train = sequences[:n_train]    # (8957, 40, 1)
y_train = results[:n_train]      # (8957, )
print(X_train.shape)
print(y_train.shape)
X_valid = sequences[n_train:-n_test]
y_valid = results[n_train:-n_test]
X_test = sequences[-n_test:]
y_test = results[-n_test:]

lr = 0.001
n_cell = 40
batch_size = 64

model = keras.models.Sequential([
    keras.layers.GRU(n_cell, input_shape=(None, 1), return_sequences=True,
                     # dropout=0.2, recurrent_dropout=0.2,
                     ),
    keras.layers.GRU(n_cell,
                     # dropout=0.2, recurrent_dropout=0.2,
                     ),
    # keras.layers.Dense(1, activation='tanh'),
    keras.layers.Dense(1),
])

s = 20 * len(X_train) // batch_size  # number of steps in 20 epochs (batch size = 32)
learning_rate = keras.optimizers.schedules.ExponentialDecay(lr, s, 0.1)
optimizer = keras.optimizers.Adam(learning_rate, clipvalue=1.0)
model.compile(loss='mae', optimizer=optimizer)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

start = time.time()
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=50,
                    batch_size=batch_size, callbacks=[early_stopping])
end = time.time()
print('Running time: %s Seconds' % (end - start))

y_pred = model.predict(X_test[:100])

actual = y_test[:100]
forecast = y_pred
time = np.linspace(0, actual.shape[0], actual.shape[0])

plt.plot(time, actual, label="Actual")
plt.plot(time, forecast, label="Forecast")
plt.legend(fontsize=14)
plt.show()