import numpy as np


# 增幅标准化
class IncreaseScaler:
    def __init__(self, input_step, output_step=1):
        self.input_step = input_step
        self.output_step = output_step
        self.denormalize_param = None

    # 用于train、valid、test，预留标签
    def train_normalize(self, train_data, step=1):
        increase = train_data[1:] / train_data[:-1] - 1  # 相较前一天的增幅

        # 总共可以切分(increase.shape[0] - input_step - output_step + 2)个样本， 每个样本(self.input_step - 1)个X时步
        data_size = increase.shape[0] - self.input_step - self.output_step + 2  # 样本总数
        batch_size = (data_size - 1) // step + 1                                # step之后批次大小
        known_step = np.empty((batch_size, self.input_step - 1))
        forecast_step = np.empty((batch_size, self.output_step))
        self.denormalize_param = np.empty((batch_size, ))    # 保存denormalize参数
        for i in range(0, data_size, step):
            self.denormalize_param[i // step] = train_data[i]
            known_step[i // step] = increase[i: i + self.input_step - 1]  # 2~input_step天相较前1天增幅
            for j in range(0, forecast_step.shape[1]):
                # 第input_step + output_step天相对第1天增幅
                forecast_step[i // step][j] = train_data[i + self.input_step + j] / train_data[i] - 1
        return known_step, forecast_step

    # 用于实际预测，不预留标签
    def predict_normalize(self, actual_data):
        assert actual_data.shape[0] == self.input_step
        increase = actual_data[1:] / actual_data[:-1] - 1

        known_step = increase.reshape((1, -1))
        self.denormalize_param = actual_data[0]
        return known_step

    def denormalize(self, forecast_step):
        assert forecast_step.shape[0] == self.denormalize_param.shape[0]
        for index in range(0, forecast_step.shape[0]):
            forecast_step[index] = (forecast_step[index] + 1) * self.denormalize_param[index]
        return forecast_step


if __name__ == '__main__':
    dataset = [i for i in range(100, 201)]
    dataset = np.array(dataset)

    scaler = IncreaseScaler(40, 1)
    X_train, Y_train = scaler.train_normalize(dataset, 3)
    print(X_train.shape)
    print(Y_train.shape)
    # print(X_train)
    # print(Y_train)
    Y_train = scaler.denormalize(Y_train)
    # print(Y_train)
    X_test = scaler.predict_normalize(dataset[:40])
    print(X_test.shape)
