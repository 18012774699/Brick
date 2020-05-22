import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


housing = fetch_california_housing()
m, n = housing.data.shape
# np.c_按colunm来组合array
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
scaler = StandardScaler()
scaler.fit(housing_data_plus_bias)
scaled_housing_data_plus_bias = scaler.transform(housing_data_plus_bias)

n_epochs = 1000
learning_rate = 0.01
t0, t1 = 5, 50  # 超参数


def learning_schedule(t):
    return t0 / (t + t1)


graph = tf.Graph()
with graph.as_default():
    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
    theta = tf.Variable(tf.compat.v1.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")

    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")

    # gradients = 2 / m * tf.matmul(tf.transpose(X), error)
    # gradients = tf.gradients(mse, [theta])[0]       # Using autodiﬀ
    # training_op = tf.compat.v1.assign(theta, theta - learning_rate * gradients)

    # 使用优化器，哪里对theta赋值
    # optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    training_op = optimizer.minimize(mse)

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session(graph=graph) as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        learning_rate = t0 / (epoch*m + t1)
        if epoch % 100 == 0:
            print("Epoch", epoch, ", MSE =", mse.eval())
            save_path = saver.save(sess, "/tmp/my_model.ckpt")
        sess.run(training_op)
    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")  # 找到tmp文件夹就找到文件了
    best_theta = theta.eval()
    print(best_theta)
