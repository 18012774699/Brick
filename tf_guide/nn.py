import tensorflow as tf

# 特征、标签配对，构建数据集
features = tf.constant([12, 23, 10, 17])
labels = tf.constant([0, 1, 1, 0])
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
print(dataset)
for element in dataset:
    print(element)
print('====================tf.data.Dataset.from_tensor_slices====================\n')

# with结构记录计算过程，
with tf.GradientTape() as tape:
    w = tf.Variable(tf.constant(3.0))
    loss = tf.pow(w, 2)
grad = tape.gradient(loss, w)
print(grad)
print('====================tf.GradientTape====================\n')
