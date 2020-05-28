import tensorflow as tf
from tensorflow import keras
from cnn_net.res_block import ResNet


def pyramid_pooling_block(input, pool_sizes):
    # print(input.shape)
    h = input.shape[1]
    w = input.shape[2]

    concat_list = [input]
    for pool_size in pool_sizes:
        x = keras.layers.AveragePooling2D(pool_size=pool_size, strides=pool_size)(input)
        print(x.shape)
        x = keras.layers.Conv2D(64, 1)(x)
        x = keras.layers.Lambda(lambda x: tf.image.resize(x, (h, w)))(x)
        concat_list.append(x)
    return keras.layers.concatenate(concat_list)


def pspnet(num_classes, input_shape=(224, 224, 3)):
    image_input = keras.layers.Input(shape=input_shape)
    x = ResNet(image_input)

    x = pyramid_pooling_block(x, [2, 4, 8, 16])
    print(x.shape)  # (None, 112, 112, 512)

    x = keras.layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Conv2D(num_classes, kernel_size=1)(x)
    x = keras.layers.Conv2DTranspose(num_classes, kernel_size=(16, 16), strides=(2, 2), padding='same')(x)
    output = keras.layers.Activation('softmax')(x)  # (None, 224, 224, 21)
    model = keras.Model(inputs=[image_input], outputs=[output])
    return model


if __name__ == '__main__':
    model = pspnet(21, input_shape=(512, 384, 3))
    print(model.summary())
