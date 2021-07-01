import numpy as np


class RatingPredMetrics(object):
    """
    评分预测的评价指标类
    """
    def __init__(self):
        super(RatingPredMetrics, self).__init__()

    def mae(self, validation_true, validation_pred):
        """
        计算平均绝对误差
        :param validation_true: 二维数组，验证数据集上的真实评分
        :param validation_pred: 二维数组，验证数据集上的预测评分
        :return: MAE的值
        """
        mask = np.where(validation_true > 0, 1, 0)
        validation_pred = validation_pred * mask
        num_scores = np.sum(mask)
        error_abs = np.abs(np.round(validation_pred) - validation_true)
        return np.sum(error_abs) / num_scores

    def rmse(self, validation_true, validation_pred):
        """
        计算均方根误差
        :param validation_true: 二维数组，验证数据集上的真实评分
        :param validation_pred: 二维数组，验证数据集上的预测评分
        :return: RMSE的值
        """
        mask = np.where(validation_true > 0, 1, 0)
        validation_pred = validation_pred * mask
        num_scores = np.sum(mask)
        error_square = np.power(np.round(validation_pred) - validation_true, 2)
        return np.sqrt(np.sum(error_square) / num_scores)
