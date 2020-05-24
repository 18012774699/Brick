import tensorflow as tf
from tensorflow import keras
from Api import img_load
from cnn_net.pspnet import pspnet


if __name__ == '__main__':
    model = pspnet()

    image_path = r"D:\AI\dataset\PASCAL VOC\ImageQrg"
    label_path = r"D:\AI\dataset\PASCAL VOC\SegmentationClass"
    image_org = img_load.read_dir(image_path)
    ground_truth = img_load.read_dir(label_path)

    dataset = tf.data.Dataset.from_tensor_slices((image_org, ground_truth)).shuffle(1000)
    dataset = dataset.batch(32, drop_remainder=True).prefetch(1)
    for image_batch, label_batch in dataset:
        print(image_batch)
        print(label_batch)

    # 对标签进行分类编码：变成one-hot编码
    train_labels_final = keras.utils.to_categorical(ground_truth)
    # mobilenet
