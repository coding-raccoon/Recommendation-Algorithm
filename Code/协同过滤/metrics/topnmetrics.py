import numpy as np


class TopNMetrics(object):
    """
    TopN推荐的评价指标计算
    """
    def __init__(self):
        super(TopNMetrics, self).__init__()

    def recall(self, test_true, test_pred):
        """
        召回率计算
        :param test_true: 测试数据真实值
        :param test_pred: 测试数据预测值
        :return: 预测召回率
        """
        return np.sum(np.where(test_pred + test_true > 1, 1, 0))/np.sum(test_true)

    def precision(self, test_true, test_pred):
        """
        准确率计算
        :param test_true: 测试数据真实值
        :param test_pred: 测试数据预测值
        :return: 预测准确率
        """
        return np.sum(np.where(test_pred + test_true > 1, 1, 0))/np.sum(test_pred)

    def coverage(self, train, test_pred):
        """
        覆盖率计算
        :param train: 训练数据
        :param test_pred: 预测数据
        :return: topN推荐覆盖率
        """
        items_all = np.sum(np.where(np.sum(train, axis=0) > 0, 1, 0))     # 计算所有的项目数
        items_pred = np.sum(np.where(np.sum(test_pred, axis=0) > 0, 1, 0))  # 计算所有topN推荐成功覆盖的项目数
        return items_pred/items_all

    def popularity(self, train, test_pred):
        """
        计算topN推荐物品的流行度
        :param train: 训练数据
        :param test_pred: 测试数据预测值
        :return: 预测物品的平均流行度
        """
        popularity_items = np.log(1 + np.sum(train, axis=0))           # 每一个物品的流行度
        popularity = np.sum(np.dot(test_pred, popularity_items.reshape(train.shape[1], 1)))      #
        return popularity/np.sum(test_pred)