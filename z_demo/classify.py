import tensorflow as tf
from tensorflow import keras
import cv2
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from Api import img_load
from cnn_net.pspnet import pspnet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

IMG_WEIGHT = 224

image_org = []

with tf.name_scope('tool'):
    # 随机裁剪，数据增强
    def rand_crop(image, label, size=(IMG_WEIGHT, IMG_WEIGHT)):
        height1 = random.randint(0, image.shape[0] - size[0])
        width1 = random.randint(0, image.shape[1] - size[1])
        height2 = height1 + size[0]
        width2 = width1 + size[1]

        image = image[height1:height2, width1:width2]
        label = label[height1:height2, width1:width2]

        assert image.shape == (IMG_WEIGHT, IMG_WEIGHT, 3)
        assert label.shape == (IMG_WEIGHT, IMG_WEIGHT, 3)
        return image, label


with tf.name_scope('img_preprocess'):
    image_org_path = r"D:\Friedrich\dataset\PASCAL VOC\ImageQrg"
    ground_truth_path = r"D:\Friedrich\dataset\PASCAL VOC\SegmentationClass"
    image_org_path = img_load.read_dir(image_org_path)
    ground_truth_path = img_load.read_dir(ground_truth_path)

    # 加载图片，过滤尺寸太小的图片
    for index in range(len(image_org_path)):
        img = cv2.imread(image_org_path[index])
        # 读图
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_org.append(img)  # image_org
        # resize
        # img = cv2.resize(img, (IMG_WEIGHT, IMG_WEIGHT), interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite(image_org_path[index], img)

    # 随机裁剪，数据增强
    # for index in range(len(ground_truth)):
    #     image_org[index], ground_truth[index] = rand_crop(image_org[index], ground_truth[index])

    image_org = np.array(image_org)
    print(image_org.shape)  # (2892, 224, 224, 3)
    print('\nimg_preprocess finished!\n====================================')

with tf.name_scope('data_preprocess'):
    # 打乱顺序
    index = [i for i in range(len(image_org))]
    random.shuffle(index)
    image_org = image_org[index]

    image_org = image_org.astype(np.float32)  # for train
    # image_org = image_org.astype(np.uint8)      # for show
    image_org = image_org / 255.0

    # 对标签进行分类编码：变成one-hot编码，loss_func = categorical_crossentropy
    # ground_truth_labels = keras.utils.to_categorical(ground_truth)
    # memory_utilization = psutil.virtual_memory().percent
    # print(ground_truth_labels.shape)  # (2892, 224, 224, 21)

    n_train = int(0.7 * len(image_org))
    n_valid = int(0.9 * len(image_org))

    X_train, Y_train = image_org[:n_train], ground_truth[:n_train]
    X_valid, Y_valid = image_org[n_train:n_valid], ground_truth[n_train:n_valid]
    X_test, Y_test = image_org[n_valid:], ground_truth[n_valid:]
    print(X_train.shape)
    print(Y_train.shape)
    print('\ndata_preprocess finished!\n====================================')


with tf.name_scope('train'):
    batch_size = 4
    lr = 0.01
    loss_func = 'sparse_categorical_crossentropy'  # 稀疏标签（0~20）

    # lr_schedule = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    save_model = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)

    s = 20 * len(X_train) // batch_size
    learning_rate = keras.optimizers.schedules.ExponentialDecay(lr, s, 0.1)
    optimizer = keras.optimizers.Adam(learning_rate, clipvalue=1.0)
    # optimizer = keras.optimizers.Adam(learning_rate)

    model = pspnet(num_classes=21, input_shape=(IMG_WEIGHT, IMG_WEIGHT, 3))
    # model = keras.models.load_model("my_keras_model.h5")
    # model.load_weights("my_keras_model.h5")

    print(model.summary())
    model.compile(loss=focal_loss, optimizer=optimizer, metrics=["sparse_categorical_accuracy"])

    start = time.time()
    history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid),
                        epochs=20, batch_size=batch_size, callbacks=[early_stopping, save_model])
    end = time.time()
    print('Trainning time: %s Seconds' % (end - start))
    print('\ntrain finished!\n====================================')

with tf.name_scope('visualization'):
    # loss_test = model.evaluate(X_test, Y_test)

    # 原图
    temp = (X_test[0]*255).astype(np.uint8)
    plt.imshow(temp)
    plt.show()
    # 标签
    y_test = label_to_pixel(Y_test[0])
    plt.imshow(y_test)
    plt.show()
    # 预测值
    temp = model.predict(X_test[:2])
    y_pred = np.argmax(temp, axis=3)
    y_pred = label_to_pixel(y_pred[0])
    plt.imshow(y_pred)
    plt.show()
    print('\nvisualization finished!\n====================================')
