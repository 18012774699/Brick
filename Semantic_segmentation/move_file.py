# 将文件copy到path
import os
import shutil


def read_dir_by_name(search_path: str):
    img_path_list = []
    for main_dir, subdir, file_name_list in os.walk(search_path):
        # print("main_dir:", main_dir)  # 当前主目录
        # print("subdir:", subdir)  # 当前主目录下的所有目录
        # print("file_name_list:", file_name_list)  # 当前主目录下的所有文件
        for filename in file_name_list:
            (file, extension) = os.path.splitext(filename)
            img_path_list.append(file)
    return img_path_list


def search_file_to_copy(search_path: str, filter_list: list):
    for main_dir, subdir, file_name_list in os.walk(search_path):
        for filename in file_name_list:
            (file, extension) = os.path.splitext(filename)
            if file not in filter_list:
                continue
            shutil.copyfile(main_dir + '/' + filename, copy_path + filename)


if __name__ == '__main__':
    image_path = r"D:\AI\dataset\PASCAL VOC\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages"
    label_path = r"D:\AI\dataset\PASCAL VOC\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\SegmentationClass"
    ground_truth_file = read_dir_by_name(label_path)    # 无后缀

    copy_path = os.path.join(os.path.expanduser('~'), "Desktop\\copy_image\\")
    search_file_to_copy(image_path, ground_truth_file)
