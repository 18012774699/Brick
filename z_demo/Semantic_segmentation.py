import tensorflow as tf
from tensorflow import keras
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from Api import img_load
from cnn_net.pspnet import pspnet

image_org = []
ground_truth = []
'''
0:background    255:ground_truth
1:aeroplane     2:bicycle   3:bird      4:boat          5:bottle
6:bus           7:car       8:cat       9:chair         10:cow
11:diningtable  12:dog      13:horse    14:motorbike    15:person
16:pottedplant  17:sheep    18:sofa     19:train        20:monitor
'''
pixel_map_to_label = {(0, 0, 0): 0, (255, 255, 255): 0,
                      (128, 0, 0): 1, (0, 128, 0): 2, (128, 128, 0): 3, (0, 0, 128): 4, (128, 0, 128): 5,
                      (0, 128, 128): 6, (128, 128, 128): 7, (64, 0, 0): 8, (192, 0, 0): 9, (64, 128, 0): 10,
                      (192, 128, 0): 11, (64, 0, 128): 12, (192, 0, 128): 13, (64, 128, 128): 14, (192, 128, 128): 15,
                      (0, 64, 0): 16, (128, 64, 0): 17, (0, 192, 0): 18, (128, 192, 0): 19, (0, 64, 128): 20}

with tf.name_scope('tool'):
    # 随机裁剪，数据增强
    def rand_crop(image, label, size=(256, 256)):
        height1 = random.randint(0, image.shape[0] - size[0])
        width1 = random.randint(0, image.shape[1] - size[1])
        height2 = height1 + size[0]
        width2 = width1 + size[1]

        image = image[height1:height2, width1:width2]
        label = label[height1:height2, width1:width2]
        assert image.shape == (256, 256, 3)
        assert label.shape == (256, 256, 3)
        return image, label


    def pixel_to_label(image):
        plt.imshow(image)
        plt.show()
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        plt.imshow(img_gray)
        plt.show()
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                print(image[i][j])
                img_gray[i][j] = pixel_map_to_label[image[i][j]]
                # if all(image[i][j] == [0, 0, 0]):
                #     img_gray[i][j] = 0
        print(image.shape)


    def label_to_pixel():
        pass

with tf.name_scope('img_preprocess'):
    '''
    image_org:
        1、加载图片
        2、resize放大较小图片
        3、随机裁剪（数据增强）
        4、像素值/255，归一化
    ground_truth:
        1~3、同上
        4、像素值转分类标签（用于训练）
        5、分类标签转像素值（用于预测结果预览）
    '''
    image_org_path = r"D:\AI\dataset\PASCAL VOC\ImageQrg"
    ground_truth_path = r"D:\AI\dataset\PASCAL VOC\SegmentationClass"
    image_org_path = img_load.read_dir(image_org_path)
    ground_truth_path = img_load.read_dir(ground_truth_path)

    # img.astype(np.float32)    # for train
    # img.astype(np.uint8)      # for show

    # 加载图片，过滤尺寸太小的图片
    for index in range(len(image_org_path)):
        img = cv2.imread(image_org_path[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if min(img.shape[0], img.shape[1]) * 2 < 256:
            continue
        image_org.append(img)  # image_org
        img = cv2.imread(ground_truth_path[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ground_truth.append(img)  # ground_truth

    # 将小于目标尺寸的图片放大一倍
    for index in range(len(ground_truth)):
        img = image_org[index]
        if img.shape[0] < 256 or img.shape[1] < 256:
            img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
            image_org[index] = img
            ground_truth[index] = cv2.resize(ground_truth[index], (img.shape[1], img.shape[0]))

    # 随机裁剪，数据增强
    for index in range(len(ground_truth)):
        image_org[index], ground_truth[index] = rand_crop(image_org[index], ground_truth[index])

    # ground_truth像素值转分类标签（用于训练）
    for img in ground_truth:
        pixel_to_label(img)

    # 打乱顺序
    index = [i for i in range(len(image_org))]
    random.shuffle(index)

    image_org = np.array(image_org)[index]
    ground_truth = np.array(ground_truth)[index]
    print(image_org.shape)  # (2912, 256, 256, 3)
    print(ground_truth.shape)  # (2912, 256, 256, 1)

# model = pspnet()

# 对标签进行分类编码：变成one-hot编码
# train_labels_final = keras.utils.to_categorical(ground_truth)
# mobilenet
