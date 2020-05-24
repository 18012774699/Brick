import os
import tensorflow as tf
from Api import img_load


def match_ground_truth(filename, filter_list):
    (file1, extension1) = os.path.splitext(filename)
    for file in filter_list:
        (file2, extension2) = os.path.splitext(file)
        if file1 == file2:
            return True
    return False


def read_file_by_name(search_path: str, filter_list: list):
    img_path_list = []
    for main_dir, subdir, file_name_list in os.walk(search_path):
        for filename in file_name_list:
            if not match_ground_truth(filename, filter_list):
                continue
            file_path = main_dir + '/' + filename
            img_path_list.append(file_path)
    return img_path_list


if __name__ == '__main__':
    image_path = r"D:\AI\dataset\PASCAL VOC\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages"
    label_path = r"D:\AI\dataset\PASCAL VOC\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\SegmentationClass"
    ground_truth = img_load.read_dir_by_name(label_path)
    image = read_file_by_name(image_path, ground_truth)
    ground_truth = [label_path + '\\' + filename for filename in ground_truth]
    train_path_set = img_load.read_dir_by_name(image_path)
    print(len(train_path_set))

    dataset = tf.data.Dataset.from_tensor_slices((train_path_set, label_train)).shuffle(1000)
    dataset = dataset.batch(32, drop_remainder=True).prefetch(1)
    for image_batch, label_batch in dataset:
        print(image_batch)
        print(label_batch)

