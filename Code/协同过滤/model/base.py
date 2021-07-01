import abc
import numpy as np
from metrics.ratingmetrics import RatingPredMetrics
from metrics.topnmetrics import TopNMetrics


class TopNBaseModel(object):
    """
    TopN推荐模型的基类，所有TopN推荐模型均直接或间接继承自该类
    属性：
    n: 需要给每个用户推荐的物品数目
    metrcs: 评价指标，topN推荐的指标有召回率、准确率、覆盖率、物品流行度
    方法：
    predidct: 预测函数，每一个子类都需要定义的方法
    evaluate: 评估方法，用于测试模型的实验
    compute_metrics: 计算指标，接收训练数据和测试数据，完成一次预测和评价指标计算
    """
    def __init__(self, N):
        """
        初始化TopN推荐模型
        :param N: 为每个用户推荐的项目数目
        """
        self.n = N
        self.metrics = TopNMetrics()

    @abc.abstractmethod
    def predict(self, *kargs, **kwargs):
        pass

    @abc.abstractmethod
    def evaluate(self, *kargs, **kwargs):
        pass

    def compute_metrics(self, data_train, data_validation, evaluation=False):
        """
        根据训练数据和测试数据完成模型预测和评价指标计算
        :param data_train: 训练数据
        :param data_validation: 测试数据
        :param evaluation:
        :return: 评估指标值
        """
        metrics = {}
        # 计算推荐矩阵
        rec_matrix = self.predict(data_train, evaluation = evaluation)
        metrics['recall'] = self.metrics.recall(data_validation, rec_matrix)
        metrics['precision'] = self.metrics.precision(data_validation, rec_matrix)
        metrics['coverage'] = self.metrics.coverage(data_train, rec_matrix)
        metrics['popularity'] = self.metrics.popularity(data_train, rec_matrix)
        return metrics


class BiGraphBaseModel(TopNBaseModel):
    """
    物质扩散模型和热传导模型的父类，实现了公共方法的部分
    """
    def __init__(self, N, steps):
        """
        初始化模型
        :param N: 每个用户需要推荐的物品数目
        :param steps: 需要扩散或传导的步数
        """
        super(BiGraphBaseModel, self).__init__(N)
        self.W = None       # 物质或温度一轮传播的转移矩阵
        self.state = None       # 当前物质或温度分布状态
        self.current_step = 0    # 当前传播步数
        self.steps = steps

    def evaluate(self, data_train, data_validation, list_steps):
        metrics = {'recall': [], 'precision': [], 'coverage': [], 'popularity': []}
        for step in list_steps:
            self.steps = step
            print(step)
            current_metrics = self.compute_metrics(data_train, data_validation, evaluation=True)
            metrics['recall'].append(current_metrics['recall'])
            metrics['precision'].append(current_metrics['precision'])
            metrics['coverage'].append(current_metrics['coverage'])
            metrics['popularity'].append(current_metrics['popularity'])
        return metrics

    @abc.abstractmethod
    def model_init(self, *kargs):
        pass

    def predict(self, data_train, evaluation=False):
        if self.W is None:
            self.model_init(data_train)
        while self.current_step < self.steps:
            self.state = np.dot(self.W, self.state)
            self.current_step += 1
        candidate_orders = self.state.T * (1 - data_train)
        topn_orders = np.argsort(candidate_orders, axis=1)[:, :self.n]  # 去掉用户已经有反馈的商品
        rec_matrix = np.zeros(data_train.shape)
        for i in range(data_train.shape[0]):
            rec_matrix[i, topn_orders[i]] = 1
        return rec_matrix


class RatingPredBaseModel(object):
    def __init__(self):
        self.metrics = RatingPredMetrics()

    @abc.abstractmethod
    def predict(self, *kargs, **kwargs):
        pass

    @abc.abstractmethod
    def evaluate(self, *kargs, **kwargs):
        pass

    def compute_metrics(self, data_train, data_validation, evaluation=False):
        """
        根据训练数据和测试数据完成模型预测和评价指标计算
        :param data_train: 训练数据
        :param data_validation: 测试数据
        :param evaluation:
        :return: 评估指标值
        """
        metrics = {}
        # 计算推荐矩阵
        rec_matrix = self.predict(data_train, evaluation=evaluation)
        metrics['mae'] = self.metrics.mae(data_validation, rec_matrix)
        metrics['rmse'] = self.metrics.rmse(data_validation, rec_matrix)
        return metrics