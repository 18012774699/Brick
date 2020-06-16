from tensorflow import keras


def identity_block(input_tensor, kernel_size, filters, activation='relu'):
    nb_filter1, nb_filter2, nb_filter3 = filters

    x = keras.layers.Conv2D(nb_filter1, 1, 1, use_bias=False)(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation)(x)

    x = keras.layers.Conv2D(nb_filter2, kernel_size, 1, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation)(x)

    x = keras.layers.Conv2D(nb_filter3, 1, 1, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Add()([x, input_tensor])
    x = keras.layers.Activation(activation)(x)
    return x


def conv_block(input_tensor, kernel_size, filters, strides=2, activation='relu'):
    nb_filter1, nb_filter2, nb_filter3 = filters

    x = keras.layers.Conv2D(nb_filter1, 1, strides=strides, use_bias=False)(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation)(x)

    x = keras.layers.Conv2D(nb_filter2, kernel_size, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation)(x)

    x = keras.layers.Conv2D(nb_filter3, 1, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)

    shortcut = keras.layers.Conv2D(nb_filter3, 1, strides=strides, use_bias=False)(input_tensor)
    shortcut = keras.layers.BatchNormalization()(shortcut)

    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.Activation(activation)(x)
    return x


def get_resnet101(image, activation='relu'):
    # Stage 1
    x = keras.layers.ZeroPadding2D(padding=(3, 3))(image)  # 0填充
    # Height/2,Width/2,64
    x = keras.layers.Conv2D(64, 7, strides=2, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    C1 = x = keras.layers.Activation(activation)(x)
    # Height/4,Width/4,64
    x = keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], strides=2, activation=activation)
    x = identity_block(x, 3, [64, 64, 256], activation=activation)
    # Height/4,Width/4,256
    C2 = x = identity_block(x, 3, [64, 64, 256], activation=activation)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], activation=activation)
    x = identity_block(x, 3, [128, 128, 512], activation=activation)
    x = identity_block(x, 3, [128, 128, 512], activation=activation)
    # Height/8,Width/8,512
    C3 = x = identity_block(x, 3, [128, 128, 512], activation=activation)
    # Stage 4
    x = conv_block(x, 3, [64, 64, 1024], activation=activation)
    block_count = 22
    for i in range(block_count):
        x = identity_block(x, 3, [64, 64, 1024], activation=activation)
    # Height/16,Width/16,1024
    C4 = x
    # Stage 5
    x = conv_block(x, 3, [32, 32, 2048], activation=activation)
    x = identity_block(x, 3, [32, 32, 2048], activation=activation)
    # Height/32,Width/32,2048
    C5 = x = identity_block(x, 3, [32, 32, 2048], activation=activation)
    return [C1, C2, C3, C4, C5]
