import numpy as np
import time
import pandas as pd
from random import normalvariate

DATA_PATH = './datasets/diabetes.csv'


# 此处不适合用标准化，因为很多数据是缺失值，应该用归一化处理
def normalization(features):
    """
    数据归一化，对每一个特征归一化处理
    """
    max_values = np.max(features, axis=0)
    min_values = np.min(features, axis=0)
    features = (features - min_values) / (max_values - min_values)
    return features


def load_data(filepath):
    table = pd.read_csv(filepath)
    data_array = table.to_numpy()
    features, labels = data_array[:, :-1], data_array[:, -1]
    features = normalization(features)
    labels = np.where(labels == 1, 1, -1)       # 原始标签为 0，1标签，此处使用 1，-1 标签，再损失函数的表达上有所不同，但是本质一样
    return features, labels


def data_iter(filepath, num_iters=5):
    features, labels = load_data(filepath)
    np.random.seed(1)
    data_numbering = np.random.randint(0, num_iters, features.shape[0])
    for i in range(num_iters):
        mask = np.where(data_numbering == i, True, False)
        train_features, val_features = features[~mask], features[mask]
        train_labels, val_labels = labels[~mask], labels[mask]
        print(len(train_labels))
        yield train_features, train_labels, val_features, val_labels


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


class FMModel(object):
    def __init__(self, lr=0.01, lambda_regulization=0.0, K=128):
        """
        模型初始化
        :param lr: 学习率
        :param lambda_regulization: 正则化系数
        :param K: 特征隐向量维度
        """
        super(FMModel, self).__init__()
        self.lr = lr
        self.K = K
        self._lambda = lambda_regulization
        self.W = None
        self.bias = 0
        self.V = None

    def fit(self, train_features, train_labels, val_features, val_labels, num_epochs=200):
        self.W = np.zeros(train_features.shape[1])
        self.V = np.random.normal(0, 0.2, (train_features.shape[1], self.K))
        for epoch in range(num_epochs):
            for i in range(train_features.shape[0]):
                interaction_1 = np.dot(train_features[i], self.V)
                tmp = train_features[i] * self.V.T
                interaction_2 = np.sum(tmp ** 2)
                interaction = 0.5 * (np.sum(interaction_1**2) - interaction_2)
                y = self.bias + np.dot(self.W, train_features[i]) + interaction
                dl_dy = (sigmoid(y * train_labels[i]) - 1) * train_labels[i]
                self.bias -= self.lr * (dl_dy + self._lambda * self.bias)
                self.W -= self.lr * (dl_dy * train_features[i] + self._lambda * self.W)
                self.V -= self.lr * (dl_dy * (np.dot(np.expand_dims(train_features[i], axis=1),
                                                     np.expand_dims(np.sum(tmp, axis=1), axis=0)) -
                                              (train_features[i]**2 * self.V.T).T) + self.V)

            if epoch > 0 and epoch % 10 == 0:
                l, acc, p, r = self.evaluation(val_features, val_labels)
                print("epoch:{}, loss:{}, accuracy:{}, precesion:{}, recall:{}".format(epoch, l, acc, p, r))
        pass

    def evaluation(self, val_features, val_labels):
        loss, acc, p, r = 0.0, 0.0, 0.0, 0.0
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(val_features.shape[0]):
            interaction_1 = np.dot(val_features[i], self.V)
            interaction_2 = np.sum(np.dot(val_features[i]**2, self.V**2))
            interaction = 0.5 * (np.dot(interaction_1, interaction_1) - interaction_2)
            y = self.bias + np.dot(self.W, val_features[i]) + interaction
            loss += -np.log(sigmoid(val_labels[i] * y)) + \
                    0.5 * self._lambda * (self.bias**2 + np.sum(self.W**2) + np.sum(self.V**2))
            y_pred = sigmoid(y)
            if y_pred > 0.5:
                if val_labels[i] == 1.0:
                    TP += 1
                else:
                    FP += 1
            else:
                if val_labels[i] == 1.0:
                    FN += 1
                else:
                    TN += 1
        acc = (TP + TN) / len(val_labels)
        p = TP / (TP + FP + 0.1)
        r = TP / (TP + FN + 0.1)
        return loss, acc, p, r


def preprocessData(data):
    feature = np.array(data.iloc[:, :-1])  # 取特征(8个特征)
    label = data.iloc[:, -1].map(lambda x: 1 if x == 1 else -1)  # 取标签并转化为 +1，-1

    # 将数组按行进行归一化
    zmax, zmin = feature.max(axis=0), feature.min(axis=0)  # 特征的最大值，特征的最小值
    feature = (feature - zmin) / (zmax - zmin)
    label = np.array(label)

    return feature, label


if __name__ == "__main__":
    trainData = './datasets/diabetes_train.txt'
    testData = './datasets/diabetes_test.txt'
    train = pd.read_csv(trainData)
    test = pd.read_csv(testData)
    dataTrain, labelTrain = preprocessData(train)
    dataTest, labelTest = preprocessData(test)
    # for dataTrain, labelTrain, dataTest, labelTest in data_iter(DATA_PATH):
    model = FMModel()
    model.fit(dataTrain, labelTrain, dataTest, labelTest)
