import tensorflow as tf
from tensorflow import keras
import cv2
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from Api import img_load
from cnn_net.pspnet import pspnet

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

TRAIN_SIZE = 224
image_org = []
ground_truth = []
'''
1、不能resize：插值计算损坏标签
2、随机裁剪，数据增强
3、显存、内存不足
'''
'''
0:background    255:ground_truth
1:aeroplane     2:bicycle   3:bird      4:boat          5:bottle
6:bus           7:car       8:cat       9:chair         10:cow
11:diningtable  12:dog      13:horse    14:motorbike    15:person
16:pottedplant  17:sheep    18:sofa     19:train        20:monitor
'''
pixel_map_to_label = {(0, 0, 0): 0, (224, 224, 192): 0,
                      (128, 0, 0): 1, (0, 128, 0): 2, (128, 128, 0): 3, (0, 0, 128): 4, (128, 0, 128): 5,
                      (0, 128, 128): 6, (128, 128, 128): 7, (64, 0, 0): 8, (192, 0, 0): 9, (64, 128, 0): 10,
                      (192, 128, 0): 11, (64, 0, 128): 12, (192, 0, 128): 13, (64, 128, 128): 14, (192, 128, 128): 15,
                      (0, 64, 0): 16, (128, 64, 0): 17, (0, 192, 0): 18, (128, 192, 0): 19, (0, 64, 128): 20}

label_map_to_pixel = dict(zip(pixel_map_to_label.values(), pixel_map_to_label.keys()))
label_map_to_pixel[0] = (0, 0, 0)

with tf.name_scope('tool'):
    # 随机裁剪，数据增强
    def rand_crop(image, label, size=(TRAIN_SIZE, TRAIN_SIZE)):
        height1 = random.randint(0, image.shape[0] - size[0])
        width1 = random.randint(0, image.shape[1] - size[1])
        height2 = height1 + size[0]
        width2 = width1 + size[1]

        image = image[height1:height2, width1:width2]
        label = label[height1:height2, width1:width2]

        assert image.shape == (TRAIN_SIZE, TRAIN_SIZE, 3)
        assert label.shape == (TRAIN_SIZE, TRAIN_SIZE, 3)
        return image, label

    # (224, 224, 3) => (224, 224,)
    def pixel_to_label(image):
        # 用灰度图生成label
        label = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pixel = tuple(image[i][j].tolist())
                label[i][j] = pixel_map_to_label[pixel]
        return label

    # (224, 224,) => (224, 224, 3)
    def label_to_pixel(label):
        label = label.reshape(TRAIN_SIZE, TRAIN_SIZE, 1)
        label = np.repeat(label, 3, axis=2)
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                label[i][j] = label_map_to_pixel[label[i][j][0]]
        return label

with tf.name_scope('img_preprocess'):
    '''
    image_org:
        1、加载图片，过滤不满足尺寸的图片
        2、随机裁剪（数据增强）
        3、像素值/255，归一化
    ground_truth:
        1~2、同上
        3、像素值转分类标签（用于训练）
        4、分类标签转像素值（用于预测结果预览）
    '''
    image_org_path = r"D:\Friedrich\dataset\PASCAL VOC\ImageQrg"
    ground_truth_path = r"D:\Friedrich\dataset\PASCAL VOC\SegmentationClass"
    image_org_path = img_load.read_dir(image_org_path)
    ground_truth_path = img_load.read_dir(ground_truth_path)

    # 加载图片，过滤尺寸太小的图片
    for index in range(len(image_org_path)):
        img = cv2.imread(image_org_path[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if min(img.shape[0], img.shape[1]) < TRAIN_SIZE:
            continue
        image_org.append(img)  # image_org
        img = cv2.imread(ground_truth_path[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ground_truth.append(img)  # ground_truth

    # 随机裁剪，数据增强
    for index in range(len(ground_truth)):
        image_org[index], ground_truth[index] = rand_crop(image_org[index], ground_truth[index])

    # ground_truth像素值转分类标签（用于训练）
    for index in range(len(ground_truth)):
        ground_truth[index] = pixel_to_label(ground_truth[index])

    image_org = np.array(image_org)
    ground_truth = np.array(ground_truth)
    print(image_org.shape)  # (2892, 224, 224, 3)
    print(ground_truth.shape)  # (2892, 224, 224,)
    print('\nimg_preprocess finished!\n====================================')

with tf.name_scope('data_preprocess'):
    # 打乱顺序
    index = [i for i in range(len(image_org))]
    random.shuffle(index)
    image_org = image_org[index]
    ground_truth = ground_truth[index]

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
    # optimizer = keras.optimizers.Adam(learning_rate, clipvalue=1.0)
    optimizer = keras.optimizers.Adam(learning_rate)

    model = pspnet(input_shape=(TRAIN_SIZE, TRAIN_SIZE, 3))
    # model = keras.models.load_model("my_keras_model.h5")
    # model.load_weights("my_keras_model.h5")

    # print(model.summary())
    model.compile(loss=loss_func, optimizer=optimizer, metrics=["accuracy"])

    start = time.time()
    history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid),
                        epochs=20, batch_size=batch_size, callbacks=[early_stopping, save_model])
    end = time.time()
    print('Trainning time: %s Seconds' % (end - start))
    print('\ntrain finished!\n====================================')

with tf.name_scope('visualization'):
    # loss_test = model.evaluate(X_test, Y_test)
    # y_pred = model.predict(X_test[0])
    print('\nvisualization finished!\n====================================')

# mobilenet
