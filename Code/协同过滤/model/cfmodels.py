import numpy as np
from .base import TopNBaseModel, RatingPredBaseModel


class ItemCFTopN(TopNBaseModel):
    """
    用于Top_N推荐的基于物品的协同过滤算法
    继承自TopN推荐模型基类
    """
    def __init__(self, N, sim_function, K=0):
        """
        初始化一个用于TopN推荐的ItemCF模型
        :param N: 给每个用户推荐的其最感兴趣的物品的数目
        :param sim_function: 计算物品之间相似度时所采用的算法，一共支持四种算法
        :param K: 邻域的大小
        """
        super(ItemCFTopN, self).__init__(N)
        self.sim_func = sim_function
        self.sim_matrix = None      # 相似度矩阵，N(items) * N(items)
        self.K = K              # 邻域大小
        self.K_eval = 0       # 用与评估的邻域大小

    def evaluate(self, data_train, data_validation, list_k):
        """
        评估函数，用于评算法的性能
        :param data_train: 二维数据，训练数据集
        :param data_validation: 二维数组，测试数据集
        :param list_k: K值变化的列表，用于测试不同的K值对算法的影响
        :return: 返回字典，包含各种指标随着超参数变化的情况
        """
        metrics = {'recall': [], 'precision': [], 'coverage': [], 'popularity': []}
        for k in list_k:
            self.K_eval = k
            current_metrics = self.compute_metrics(data_train, data_validation, evaluation=True)
            metrics['recall'].append(current_metrics['recall'])
            metrics['precision'].append(current_metrics['precision'])
            metrics['coverage'].append(current_metrics['coverage'])
            metrics['popularity'].append(current_metrics['popularity'])
        return metrics

    def predict(self, train_data, evaluation=False):
        """
        预测函数，根据训练数据，预测针对每一个用户的topN推荐
        :param train_data: 二维数组，训练数据，train_data[i][j]表示用户i对物品j是否有反馈
        :param evaluation: 是否是评估模型
        :return: topN推荐矩阵
        """
        k_nn = self.K_eval
        if not evaluation:
            k_nn = self.K
            self.sim_matrix = None
        # 如过相似度矩阵没有被计算，需要计算相似度矩阵
        if self.sim_matrix is None:
            self.sim_matrix = self.sim_func(train_data.T)
        k_neghbors = np.argsort(-self.sim_matrix, axis=1)[:, :k_nn]    # 找到最相似度的K个物品
        mask = np.zeros(self.sim_matrix.shape)        # 生成邻域mask
        for i in range(self.sim_matrix.shape[0]):
            mask[i, k_neghbors[i]] = 1.0
        sim_neghbors = mask * self.sim_matrix         # 只考虑领每个物品与邻域中物品的相似度
        item_rate = np.dot(train_data, sim_neghbors)   # 根据邻域计算出每个用户对每件物品的兴趣度
        cadidate_items_rate = item_rate * (1-train_data)    # 去掉用户已经有反馈的物品
        topn_orders = np.argsort(-cadidate_items_rate, axis=1)[:, :self.n]      # 每个用户的topN推荐物品号
        rec_matrix = np.zeros(train_data.shape)       # 生成推荐矩阵
        for i in range(train_data.shape[0]):
            rec_matrix[i, topn_orders[i]] = 1
        return rec_matrix


class UserCFTopN(TopNBaseModel):
    """
    用于Top_N推荐的基于用户的协同过滤算法
    继承自TopN推荐模型基类
    """
    def __init__(self, N, sim_function, K=0):
        """
        初始化一个用于TopN推荐的ItemCF模型
        :param N: 给每个用户推荐的其最感兴趣的物品的数目
        :param sim_function: 计算用户之间相似度时所采用的算法，一共支持四种算法
        :param K: 邻域的大小
        """
        super(UserCFTopN, self).__init__(N)
        self.sim_func = sim_function     # 用户相似度计算函数
        self.sim_matrix = None    # 用户相似度矩阵
        self.K = K     # 邻域大小
        self.K_eval = 0

    def evaluate(self, data_train, data_validation, list_k):
        """
        评估函数，用于邻域大小变化对算法性能的影响
        :param data_train: 二维数据，训练数据集
        :param data_validation: 二维数组，测试数据集
        :param list_k: K值变化的列表，用于测试不同的K值对算法的影响
        :return: 返回字典，包含各种指标随着超参数变化的情况
        """
        self.sim_matrix = None
        metrics = {'recall': [], 'precision': [], 'coverage': [], 'popularity': []}
        for k in list_k:
            self.K_eval = k
            current_metrics = self.compute_metrics(data_train, data_validation, evaluation=True)
            metrics['recall'].append(current_metrics['recall'])
            metrics['precision'].append(current_metrics['precision'])
            metrics['coverage'].append(current_metrics['coverage'])
            metrics['popularity'].append(current_metrics['popularity'])
        return metrics

    def predict(self, train_data, evaluation=False):
        """
        预测函数，根据训练数据，预测针对每一个用户的topN推荐
        :param train_data: 二维数组，训练数据，train_data[i][j]表示用户i对物品j是否有反馈
        :param evaluation: 是否是评估模型
        :return: topN推荐矩阵
        """
        k_nn = self.K_eval
        if not evaluation:
            k_nn = self.K
            self.sim_matrix = None
        if self.sim_matrix is None:
            self.sim_matrix = self.sim_func(train_data)
        k_neghbors = np.argsort(-self.sim_matrix, axis=1)[:, :k_nn]       # 根据相似度矩阵，得到每个用户的K邻域
        mask = np.zeros(self.sim_matrix.shape)          # 生成邻域mask
        for i in range(self.sim_matrix.shape[0]):
            mask[i, k_neghbors[i]] = 1.0
        sim_neghbors = mask * self.sim_matrix        # 只考虑每个用户和其K邻域之中的用户的相似度
        item_rate = np.dot(sim_neghbors, train_data)       # 根据邻域，计算每个用户对每件商品的兴趣度
        cadidate_items_rate = item_rate * (1-train_data)       # 去掉用户已经有反馈的商品
        topn_orders = np.argsort(-cadidate_items_rate, axis=1)[:, :self.n]
        rec_matrix = np.zeros(train_data.shape)
        for i in range(train_data.shape[0]):
            rec_matrix[i, topn_orders[i]] = 1
        return rec_matrix


class UserCFRating(RatingPredBaseModel):
    def __init__(self, sim_function, K=0):
        super(UserCFRating, self).__init__()
        self.sim_matrix = None
        self.sim_func = sim_function
        self.K = K
        self.K_eval = 0

    def evaluate(self, data_train, data_validation, list_k):
        """
        评估函数，用于邻域大小变化对算法性能的影响
        :param data_train: 二维数据，训练数据集
        :param data_validation: 二维数组，测试数据集
        :param list_k: K值变化的列表，用于测试不同的K值对算法的影响
        :return: 返回字典，包含各种指标随着超参数变化的情况
        """
        self.sim_matrix = None
        metrics = {'mae': [], 'rmse': []}
        for k in list_k:
            self.K_eval = k
            current_metrics = self.compute_metrics(data_train, data_validation, evaluation=True)
            metrics['mae'].append(current_metrics['mae'])
            metrics['rmse'].append(current_metrics['rmse'])
        return metrics

    def predict(self, train_data, evaluation=False):
        """
        预测函数，根据训练数据，预测针对每一个用户的topN推荐
        :param train_data: 二维数组，训练数据，train_data[i][j]表示用户i对物品j是否有反馈
        :param evaluation: 是否是评估模型
        :return: topN推荐矩阵
        """
        k_nn = self.K_eval
        if not evaluation:
            k_nn = self.K
            self.sim_matrix = None
        if self.sim_matrix is None:
            self.sim_matrix = self.sim_func(train_data)
        k_neghbors = np.argsort(-self.sim_matrix, axis=1)[:, :k_nn]       # 根据相似度矩阵，得到每个用户的K邻域
        sim_mask = np.zeros(self.sim_matrix.shape)          # 生成邻域mask
        for i in range(self.sim_matrix.shape[0]):
            sim_mask[i, k_neghbors[i]] = 1.0
        sim_neghbors = sim_mask * self.sim_matrix        # 只考虑每个用户和其K邻域之中的用户的相似度
        data_mask = np.where(train_data > 0, 1.0, 0)
        sum_scores = np.sum(train_data, axis=1)
        mean_scores = sum_scores / np.sum(data_mask, axis=1)        # 计算每个用户的平均打分
        zero_center_data = (train_data.T - mean_scores).T * data_mask      # 计算去掉平均打分后的评分
        weighted_error = np.dot(sim_neghbors, zero_center_data)     # 计算加权评分误差
        abs_weights_sum = np.dot(np.abs(sim_neghbors), data_mask)       # 计算权重和
        item_rate = ((weighted_error / np.where(abs_weights_sum != 0, abs_weights_sum, 0.1)).T + mean_scores).T        # 根据邻域，计算每个用户对每件商品的打分
        return item_rate * (1 - data_mask)


class ItemCFRating(RatingPredBaseModel):
    def __init__(self, sim_function, K=0):
        super(ItemCFRating, self).__init__()
        self.sim_matrix = None
        self.sim_func = sim_function
        self.K = K
        self.K_eval = 0

    def evaluate(self, data_train, data_validation, list_k):
        """
        评估函数，用于邻域大小变化对算法性能的影响
        :param data_train: 二维数据，训练数据集
        :param data_validation: 二维数组，测试数据集
        :param list_k: K值变化的列表，用于测试不同的K值对算法的影响
        :return: 返回字典，包含各种指标随着超参数变化的情况
        """
        self.sim_matrix = None
        metrics = {'mae': [], 'rmse': []}
        for k in list_k:
            self.K_eval = k
            current_metrics = self.compute_metrics(data_train, data_validation, evaluation=True)
            metrics['mae'].append(current_metrics['mae'])
            metrics['rmse'].append(current_metrics['rmse'])
        return metrics

    def predict(self, train_data, evaluation=False):
        """
        预测函数，根据训练数据，预测针对每一个用户的推荐矩阵
        :param train_data: 二维数组，训练数据，train_data[i][j]表示用户i对物品j是否有反馈
        :param evaluation: 是否是评估模型
        :return: 预测的评分矩阵
        """
        k_nn = self.K_eval
        if not evaluation:
            k_nn = self.K
            self.sim_matrix = None
        if self.sim_matrix is None:
            self.sim_matrix = self.sim_func(train_data.T)
        k_neghbors = np.argsort(-self.sim_matrix, axis=1)[:, :k_nn]       # 根据相似度矩阵，得到每件商品的K邻域
        sim_mask = np.zeros(self.sim_matrix.shape)          # 生成邻域mask
        for i in range(self.sim_matrix.shape[0]):
            sim_mask[i, k_neghbors[i]] = 1.0
        sim_neghbors = sim_mask * self.sim_matrix        # 只考虑每个商品和其K邻域之中的用户的相似度
        data_mask = np.where(train_data > 0, 1.0, 0)
        weighted_score_sum = np.dot(train_data, sim_neghbors)
        abs_weights_sum = np.dot(data_mask, np.abs(sim_neghbors))
        item_rate = weighted_score_sum / np.where(abs_weights_sum != 0, abs_weights_sum, 0.1)  # 根据邻域，计算每个用户对每件商品的打分
        return item_rate * (1 - data_mask)