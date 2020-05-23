# 从图片中提取人脸
import os
import cv2
import time
import shutil

classifier_path = r"D:/Friedrich/anaconda3/envs/tensorflow/Lib/site-packages/cv2/data/"


def get_all_image_path(dir_path, mask=None):
    if mask is None:
        mask = ['.jpg', '.JPG', '.png', '.PNG']
    image_path_list = []
    for main_dir, subdir, file_name_list in os.walk(dir_path):
        # print("main_dir:", main_dir)  # 当前主目录
        # print("subdir:", subdir)  # 当前主目录下的所有目录
        # print("file_name_list:", file_name_list)  # 当前主目录下的所有文件
        for filename in file_name_list:
            if os.path.splitext(filename)[1] in mask:
                file_path = os.path.join(main_dir, filename)
                image_path_list.append(file_path)
    return image_path_list


# 从源路径中读取所有图片放入一个list，然后逐一进行检查，把其中的脸扣下来，存储到目标路径中
def extract_face_from_images(src_path, face_path, invalid_path, classifier_type, mask=None):
    image_path_list = get_all_image_path(src_path, mask)

    # 对list中图片逐一进行检查,找出其中的人脸然后写到目标文件夹下
    count = 0
    error_count = 0
    face_detector = cv2.CascadeClassifier(classifier_path + classifier_type)
    for imagePath in image_path_list:
        try:
            img = cv2.imread(imagePath)
            if type(img) != str:
                faces = face_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3,
                                                       minSize=(50, 50), maxSize=(200, 200),
                                                       flags=cv2.CASCADE_SCALE_IMAGE)
                if len(faces):
                    for (x, y, width, height) in faces:
                        # 设置人脸宽度大于128像素，去除较小的人脸
                        if width >= 128 and height >= 128:
                            # 以时间戳和读取的排序作为文件名称
                            file_name = ''.join(str(int(time.time())) + str(count))
                            face = img[y:y + height, x:x + width]
                            face = cv2.resize(face, (128, 128))
                            cv2.imwrite(face_path + os.sep + '%s.jpg' % file_name, face)
                            count += 1
                            print(imagePath + "have face")
                else:
                    shutil.move(imagePath, invalid_path)  # 移动错误图片
        except IOError:
            error_count += 1
            print("Error")
            continue
    print('Find ' + str(count) + ' faces to Destination ' + face_path)
    print('error_count: ' + str(error_count))


if __name__ == '__main__':
    sourcePath = r'C:\Users\d84138318\Desktop\originalPics'
    targetPath = r'C:\Users\d84138318\Desktop\face'
    invalidPath = r'C:\Users\d84138318\Desktop\NoPeaple'
    classifier_type = "haarcascade_frontalface_alt.xml"
    extract_face_from_images(sourcePath, targetPath, invalidPath, classifier_type)
