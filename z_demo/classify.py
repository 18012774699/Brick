import tensorflow as tf
from tensorflow import keras
import cv2
import random
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from cnn_net.res_block import ResNet
from cnn_net.resnet101 import get_resnet101
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

IMG_SIZE = 224
image = []

with tf.name_scope('tool'):
    # 随机裁剪，数据增强
    def rand_crop(image, label, size=(IMG_SIZE, IMG_SIZE)):
        height1 = random.randint(0, image.shape[0] - size[0])
        width1 = random.randint(0, image.shape[1] - size[1])
        height2 = height1 + size[0]
        width2 = width1 + size[1]

        image = image[height1:height2, width1:width2]
        label = label[height1:height2, width1:width2]

        assert image.shape == (IMG_SIZE, IMG_SIZE, 3)
        assert label.shape == (IMG_SIZE, IMG_SIZE, 3)
        return image, label


with tf.name_scope('img_preprocess'):
    label_path = r"E:\dataset\Kaggle\aptos2019-blindness-detection\train.csv"
    label_data = pd.read_csv(label_path)
    image_name = label_data["id_code"].values
    label = label_data["diagnosis"].values
    image_path = r"E:\dataset\Kaggle\aptos2019-blindness-detection\train_images/" + image_name + ".png"

    # 加载图片，过滤尺寸太小的图片
    for index in range(len(image_path)):
        img = cv2.imread(image_path[index])
        # 读图
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.append(img)  # image
        # resize
        # img_height = img.shape[0]
        # img_weight = img.shape[1]
        # if img_height > img_weight:
        #     top_size = 0
        #     bottom_size = 0
        #     left_size = (img_height - img_weight) // 2
        #     right_size = (img_height - img_weight) // 2
        #     img = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,
        #                              cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # elif img_height < img_weight:
        #     left_size = 0
        #     right_size = 0
        #     top_size = (img_weight - img_height) // 2
        #     bottom_size = (img_weight - img_height) // 2
        #     img = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,
        #                              cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite(image_path[index], img)

    image = np.array(image)
    print(image.shape)  # (3662, 224, 224, 3)
    print('\nimg_preprocess finished!\n====================================')

with tf.name_scope('data_preprocess'):
    # 打乱顺序
    index = [i for i in range(len(image))]
    random.shuffle(index)
    image = image[index]
    label = label[index]

    image = image.astype(np.float32)  # for train
    # image = image.astype(np.uint8)      # for show
    image = image / 255.0

    # 对标签进行分类编码：变成one-hot编码，loss_func = categorical_crossentropy
    # ground_truth_labels = keras.utils.to_categorical(ground_truth)
    # memory_utilization = psutil.virtual_memory().percent
    # print(ground_truth_labels.shape)

    n_train = int(0.7 * len(image))

    X_train, Y_train = image[:n_train], label[:n_train]
    X_valid, Y_valid = image[n_train:], label[n_train:]
    print(X_train.shape)
    print(Y_train.shape)
    print('\ndata_preprocess finished!\n====================================')


with tf.name_scope('train'):
    batch_size = 32
    lr = 1e-1
    loss_func = 'sparse_categorical_crossentropy'  # 稀疏标签（0~20）
    num_classes = 5

    # lr_schedule = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    save_model = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True, save_weights_only=True)

    s = 20 * len(X_train) // batch_size
    learning_rate = keras.optimizers.schedules.ExponentialDecay(lr, s, 0.1)
    optimizer = keras.optimizers.Adam(learning_rate, clipvalue=1.0)
    # optimizer = keras.optimizers.Adam(learning_rate)

    image_input = keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    C1, C2, C3, C4, C5 = get_resnet101(image_input, activation='relu')
    x = C5
    x = keras.layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    output = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=[image_input], outputs=[output])
    # model = keras.models.load_model("my_keras_model.h5")
    model.load_weights("my_keras_model.h5")

    print(model.summary())
    model.compile(loss=loss_func, optimizer=optimizer, metrics=["accuracy"])

    start = time.time()
    history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid),
                        epochs=100, batch_size=batch_size, callbacks=[early_stopping, save_model])
    end = time.time()
    print('Trainning time: %s Seconds' % (end - start))
    print('\ntrain finished!\n====================================')

with tf.name_scope('visualization'):
    label_path = r"D:\Friedrich\dataset\Kaggle\aptos2019-blindness-detection\test.csv"
    label_data = pd.read_csv(label_path)
    image_name = label_data["id_code"].values
    image_path = r"D:\Friedrich\dataset\Kaggle\aptos2019-blindness-detection\test_images/" + image_name + ".png"

    test_img = []
    for index in range(len(image_path)):
        img = cv2.imread(image_path[index])
        # 读图
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_img.append(img)  # image

    test_img = np.array(test_img)
    test_img = test_img.astype(np.float32)  # for train
    test_img = test_img / 255.0

    res = model.predict(test_img)
    print(res)
    print('\nvisualization finished!\n====================================')
