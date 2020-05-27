from tensorflow import keras
from cnn_net.resnet101 import get_resnet101


def pspnet(input_shape=(224, 224, 3)):
    image_input = keras.layers.Input(shape=input_shape)
    output_categories = 21
    layers = get_resnet101(image_input)

    UpSample = []
    for layer in layers:
        layer = keras.layers.Conv2D(128, 1, use_bias=False)(layer)
        layer = keras.layers.BatchNormalization()(layer)
        layer = keras.layers.Activation('relu')(layer)
        print(layer.shape)
        if layer.shape[1] != input_shape[0]:
            size = input_shape[0] / layer.shape[1]
            layer = keras.layers.UpSampling2D(size=(size, size))(layer)
        UpSample.append(layer)
        # print(layer.shape)
    concat = keras.layers.concatenate(UpSample)  # (None, 256, 256, 640)
    x = keras.layers.Conv2D(100, 3, padding='same', use_bias=False)(concat)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(output_categories, 1)(x)
    output = keras.layers.Activation('softmax')(x)  # (None, 256, 256, 21)

    model = keras.Model(inputs=[image_input], outputs=[output])
    return model


if __name__ == '__main__':
    model = pspnet(input_shape=(224, 224, 3))
    print(model.summary())
