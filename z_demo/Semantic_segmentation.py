import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
from Api import img_load
# from cnn_net.pspnet import pspnet


# 1、加载图片
# 2、resize
# 3、将ground_truth像素值替换为分类标签

if __name__ == '__main__':
    # model = pspnet()

    image_path = r"D:\AI\dataset\PASCAL VOC\ImageQrg"
    label_path = r"D:\AI\dataset\PASCAL VOC\SegmentationClass"
    image_org = img_load.read_dir(image_path)
    ground_truth = img_load.read_dir(label_path)

    # 对标签进行分类编码：变成one-hot编码
    # train_labels_final = keras.utils.to_categorical(ground_truth)
    # mobilenet

    img = plt.imread(image_org[0])
    # img.astype(np.float32)    # for train
    # img.astype(np.uint8)      # for show
    plt.imshow(img)
    plt.show()

    img = cv2.imread(image_org)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
