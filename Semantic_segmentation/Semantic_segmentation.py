import tensorflow as tf
from Api import img_load


if __name__ == '__main__':
    image_path = r"D:\AI\dataset\PASCAL VOC\ImageQrg"
    label_path = r"D:\AI\dataset\PASCAL VOC\SegmentationClass"
    image_org = img_load.read_dir(image_path)
    ground_truth = img_load.read_dir(label_path)

    dataset = tf.data.Dataset.from_tensor_slices((train_path_set, label_train)).shuffle(1000)
    dataset = dataset.batch(32, drop_remainder=True).prefetch(1)
    for image_batch, label_batch in dataset:
        print(image_batch)
        print(label_batch)

