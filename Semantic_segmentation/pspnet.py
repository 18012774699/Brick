from tensorflow import keras
from CNN.resnet101 import get_resnet101


def pspnet():
    input_shape = (256, 256, 3)
    image_input = keras.layers.Input(shape=input_shape)
    output_categories = 21
    layers = get_resnet101(image_input)

    UpSample = []
    for layer in layers:
        layer = keras.layers.Conv2D(128, 1, use_bias=False)(layer)
        layer = keras.layers.BatchNormalization()(layer)
        layer = keras.layers.Activation('relu')(layer)
        # print(layer.shape)
        if layer.shape[1] != input_shape[0]:
            size = input_shape[0]/layer.shape[1]
            layer = keras.layers.UpSampling2D(size=(size, size))(layer)
        UpSample.append(layer)
        # print(layer.shape)
    concat = keras.layers.concatenate(UpSample)     # (None, 256, 256, 640)
    x = keras.layers.Conv2D(output_categories, 1, use_bias=False)(concat)
    x = keras.layers.BatchNormalization()(x)
    output = keras.layers.Activation('softmax')(x)  # (None, 256, 256, 21)

    model = keras.Model(inputs=[image_input], outputs=[output])
    return model


model = pspnet()
print(model.summary())
# model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
# history = model.fit(X_train_A, y_train, epochs=20, validation_data=(X_valid_A, y_valid))
# mse_test = model.evaluate(X_test_A, y_test)
# y_pred = model.predict(X_new_A)
