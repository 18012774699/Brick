from tensorflow import keras


def identity_block(input_tensor, kernel_size, filters):
    nb_filter1, nb_filter2, nb_filter3 = filters

    x = keras.layers.Conv2D(nb_filter1, 1, 1, use_bias=False)(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(nb_filter2, kernel_size, 1, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(nb_filter3, 1, 1, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Add()([x, input_tensor])
    x = keras.layers.Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, strides=2):
    nb_filter1, nb_filter2, nb_filter3 = filters

    x = keras.layers.Conv2D(nb_filter1, 1, strides=strides, use_bias=False)(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(nb_filter2, kernel_size, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(nb_filter3, 1, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)

    shortcut = keras.layers.Conv2D(nb_filter3, 1, strides=strides, use_bias=False)(input_tensor)
    shortcut = keras.layers.BatchNormalization()(shortcut)

    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    return x


def ResNet(image):
    x = keras.layers.ZeroPadding2D((3, 3))(image)
    x = keras.layers.Conv2D(64, (7, 7))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = conv_block(x, 3, [64, 64, 256])
    x = identity_block(x, 3, [64, 64, 256])
    x = identity_block(x, 3, [64, 64, 256])
    x = identity_block(x, 3, [64, 64, 256])
    x = identity_block(x, 3, [64, 64, 256])
    x = identity_block(x, 3, [64, 64, 256])
    return x
