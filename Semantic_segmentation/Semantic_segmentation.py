import tensorflow as tf
from Api import img_load


def mark_dataset(data_path_set):
    label_set = []
    for path in data_path_set:
        if path.find('cats/cat') > -1:
            label_set.append(0)
        else:
            label_set.append(1)
    return data_path_set, label_set


if __name__ == '__main__':
    image_path = r"D:\AI\dataset\图像\dogs-cats-images\dataset\training_set"
    label_path = r"D:\AI\dataset\PASCAL VOC\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\SegmentationClass"
    ground_truth = img_load.read_dir_by_name(label_path)
    ground_truth = [label_path + '\\' + filename for filename in ground_truth]
    train_path_set = img_load.read_dir_by_name(image_path)
    print(len(train_path_set))

    train_path_set, label_train = mark_dataset(image_path + train_path_set)
    test_path_set, label_test = mark_dataset(test_path_set)

    dataset = tf.data.Dataset.from_tensor_slices((train_path_set, label_train)).shuffle(1000)
    dataset = dataset.batch(32, drop_remainder=True).prefetch(1)
    for image_batch, label_batch in dataset:
        print(image_batch)
        print(label_batch)

