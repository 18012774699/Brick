from tensorflow import keras
from CNN.resnet101 import get_resnet101


def pspnet():
    shape = (512, 512, 3)
    layers = get_resnet101(shape)

    UpSample = []
    for layer in layers:
        layer = keras.layers.Conv2D(128, 1, use_bias=False)(layer)
        layer = keras.layers.BatchNormalization()(layer)
        layer = keras.layers.Activation('relu')(layer)
        # print(layer.shape[1])
        if layer.shape[1] != 256:
            layer = keras.layers.UpSampling2D(size=(256/layer.shape[1], 256/layer.shape[1]))(layer)
        UpSample.append(layer)
        # print(layer.shape[1])


pspnet()
