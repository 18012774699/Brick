import os
import cv2
import matplotlib.pyplot as plt


def match_extension(file_name: str, file_type: list) -> bool:
    # (filepath, tempfilename) = os.path.split(file_name)
    (filename, extension) = os.path.splitext(file_name)
    return extension in file_type


def read_dir(search_path: str, file_type: list = [".png", ".jpg"]):
    img_path_list = []
    for main_dir, subdir, file_name_list in os.walk(search_path):
        # print("main_dir:", main_dir)  # 当前主目录
        # print("subdir:", subdir)  # 当前主目录下的所有目录
        # print("file_name_list:", file_name_list)  # 当前主目录下的所有文件
        for filename in file_name_list:
            if not match_extension(filename, file_type):
                continue
            file_path = main_dir + '/' + filename
            img_path_list.append(file_path)
    return img_path_list


if __name__ == '__main__':
    test_path = r"D:\AI\dataset\PASCAL VOC\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\SegmentationClass"
    test_set = read_dir(test_path)
    # image = cv2.cvtColor(test_set[0], cv2.COLOR_BGR2RGB)
    img = plt.imread(test_set[0])

    # img.astype(np.float32)    # for train
    # img.astype(np.uint8)      # for show
    plt.imshow(img)
    plt.show()
