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


# 将image和label转为list格式数据，因为后边用到的一些tensorflow函数接收的是list格式数据
# batch_size：每个batch要放多少张图片
# capacity：一个队列最大多少
def get_batch(image_path, label, image_size, batch_size, capacity):
    # tf.cast()用来做类型转换
    image = tf.cast(image_path, tf.string)  # 可变长度的字节数组.每一个张量元素都是一个字节数组
    label = tf.cast(label, tf.int32)
    # tf.train.slice_input_producer是一个tensor生成器
    # 作用是按照设定，每次从一个tensor列表中按顺序或者随机抽取出一个tensor放入文件名队列。
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # 读取图片

    # step2：将图像解码，使用相同类型的图像
    image = tf.image.decode_jpeg(image_contents, channels=3)
    # jpeg或者jpg格式都用decode_jpeg函数，其他格式可以去查看官方文档

    # step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
    # 标准化处理
    image = tf.image.per_image_standardization(image)

    # step4：生成batch
    # image_batch: 4D tensor [batch_size, width, height, 3], dtype = tf.float32
    # label_batch: 1D tensor [batch_size], dtype = tf.int32
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=16, capacity=capacity)

    label_batch = tf.reshape(label_batch, [batch_size])
    # image_batch = tf.cast(image_batch, tf.uint8)    # 显示彩色图像
    image_batch = tf.cast(image_batch, tf.float32)  # 显示灰度图
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

    image_batch, label_batch = get_batch(train_path_set, )

