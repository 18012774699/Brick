# 随机梯度下降
import numpy as np

X0 = np.ones((100, 1))
X1 = 2 * np.random.rand(100, 1)     # 随机样本因素
X = np.c_[X0, X1]
y = 9 + 5 * X1 + np.random.randn(100, 1)
# print(X)

t0, t1 = 5, 50  # 超参数
n_epochs = 500  # 迭代次数
m = 1000        # 样本数


def learning_schedule(t):
    return t0 / (t + t1)


theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        learning_rate = learning_schedule(epoch * m + i)
        theta = theta - learning_rate * gradients
    # if epoch % 50:
    #     print(theta)

print(theta)
