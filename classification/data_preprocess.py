from Api import img_io

train_path = r"D:\AI\dataset\图像\dogs-cats-images\dataset\training_set"
test_path = r"D:\AI\dataset\图像\dogs-cats-images\dataset\test_set"

train_path_set = img_io.read_dir(train_path)
test_path_set = img_io.read_dir(test_path)
print(len(train_path_set))

label_train = []
label_test = []

for path in train_path_set:
    if path.find('cats/cat') > -1:
        label_train.append(0)
    else:
        label_train.append(1)

for path in test_path_set:
    if path.find('cats/cat') > -1:
        label_test.append(0)
    else:
        label_test.append(1)

