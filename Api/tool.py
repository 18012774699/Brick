import os
import cv2


def match_extension(file_name: str, file_type: list) -> bool:
    # (filepath, tempfilename) = os.path.split(file_name)
    (filename, extension) = os.path.splitext(file_name)
    return extension in file_type


def read_dir(search_path: str, file_type: list):
    img_list = []
    for main_dir, subdir, file_name_list in os.walk(search_path):
        # print("main_dir:", main_dir)  # 当前主目录
        # print("subdir:", subdir)  # 当前主目录下的所有目录
        # print("file_name_list:", file_name_list)  # 当前主目录下的所有文件
        for filename in file_name_list:
            if not match_extension(filename, file_type):
                continue
            file_path = main_dir + '/' + filename
            img = cv2.imread(file_path)
            img_list.append(img)
    return img_list
