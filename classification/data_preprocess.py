import tensorflow as tf
from Api import img_io


def mark_dataset(data_path_set):
    label_set = []
    for path in data_path_set:
        if path.find('cats/cat') > -1:
            label_set.append(0)
        else:
            label_set.append(1)
    return data_path_set, label_set


def preprocess(image_path, label_batch):
    # tf.cast()用来做类型转换
    image_path = tf.cast(image_path, tf.string)
    label_batch = tf.cast(label_batch, tf.int32)

    # 读取图片
    image_contents = tf.gfile.FastGFile(image_path, 'rb').read()

    # step2：将图像解码，使用相同类型的图像
    image = tf.image.decode_jpeg(image_contents, channels=3)
    # jpeg或者jpg格式都用decode_jpeg函数，其他格式可以去查看官方文档

    # step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    image_size = 256
    image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
    # 标准化处理
    image = tf.image.per_image_standardization(image)

    # image_batch = tf.cast(image_batch, tf.uint8)    # 显示彩色图像
    image_batch = tf.cast(image, tf.float32)  # 显示灰度图
    # print(label_batch) Tensor("Reshape:0", shape=(6,), dtype=int32)
    return image_batch, label_batch


if __name__ == '__main__':
    train_path = r"D:\AI\dataset\图像\dogs-cats-images\dataset\training_set"
    test_path = r"D:\AI\dataset\图像\dogs-cats-images\dataset\test_set"
    train_path_set = img_io.read_dir(train_path)
    # test_path_set = img_io.read_dir(test_path)
    print(len(train_path_set))

    train_path_set, label_train = mark_dataset(train_path_set)
    # test_path_set, label_test = mark_dataset(test_path_set)

    dataset = tf.data.Dataset.from_tensor_slices((train_path_set, label_train)).shuffle(1000).map(preprocess)
    dataset = dataset.batch(32, drop_remainder=True).prefetch(1)
    for image_batch, label_batch in dataset:
        print(image_batch)
        print(label_batch)

