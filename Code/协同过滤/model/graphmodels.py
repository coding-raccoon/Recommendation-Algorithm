import numpy as np
from .base import BiGraphBaseModel


class MaterialDiffusionTopN(BiGraphBaseModel):
    """
    基于二部图的物质扩散算法
    继承自基于二部图推荐算法的基类BiGraphBaseModel
    """
    def __init__(self, N, steps=0):
        """
        初始化物质扩散算法
        :param N: 为每个用户推荐的最感兴趣的物品数目
        :param steps: 物质扩散的轮数，在图中一轮扩散包括两步，先从项目扩散到用户，再从用户扩散到项目
        """
        super(MaterialDiffusionTopN, self).__init__(N, steps)

    def model_init(self, data_train):
        """
        模型参数的初始化，包括初始化物质的分布状态和物质转移矩阵
        :param data_train: 二维数组，训练数据
        """
        self.state = data_train.T        # 初始物质分布状态，和用户对物品的反馈相同
        sum_row = np.sum(data_train, axis=0)
        sum_col = np.sum(data_train, axis=1)
        sum_row = np.where(sum_row > 0, sum_row, 1.0)
        sum_col = np.where(sum_col > 0, sum_col, 1.0)
        np.sum(data_train, axis=1)
        iu_trans = data_train / sum_row
        ui_trans = self.state / sum_col
        self.W = np.dot(ui_trans, iu_trans)


class HeatConductionTopN(BiGraphBaseModel):
    """
    基于二部图的热传导算法
    继承自基于二部图推荐算法的基类BiGraphBaseModel
    """
    def __init__(self, N, steps=0):
        """
        初始化热传导算法
        :param N: 为每个用户推荐的最感兴趣的物品数目
        :param steps: 热传导的轮数，在图中一轮传输包括两步，先从项目传输到用户，再从用户传输到项目
        """
        super(HeatConductionTopN, self).__init__(N, steps)

    def model_init(self, data_train):
        self.state = data_train.T
        sum_row = np.sum(data_train, axis=0)
        sum_col = np.sum(data_train, axis=1)
        sum_row = np.where(sum_row > 0, sum_row, 1.0)
        sum_col = np.where(sum_col > 0, sum_col, 1.0)
        iu_trans = (data_train.T / sum_col).T
        ui_trans = (data_train / sum_row).T
        self.W = np.dot(ui_trans, iu_trans)


class HybridBiGraphTopN(MaterialDiffusionTopN):
    """
    物质扩散算法和热传导算法的结合
    """
    def __init__(self, N, lamb1=0.5, lamb2=0.5, steps=0):
        """
        初始化
        :param N: 为每个用户推荐的最安兴的项目
        :param lamb: 决定各个模型的比重的参数
        :param steps: 传导的轮数
        """
        super(MaterialDiffusionTopN, self).__init__(N, steps)
        self.lamb1 = lamb1
        self.lamb2 = lamb2

    def model_init(self, data_train):
        self.state = data_train.T
        sum_row = np.sum(data_train, axis=0)
        sum_col = np.sum(data_train, axis=1)
        sum_row = np.where(sum_row > 0, sum_row, 1.0)
        sum_col = np.where(sum_col > 0, sum_col, 1.0)
        w1 = np.dot(data_train.T, data_train/sum_row)
        w = (data_train.T / sum_col).T
        w2 = np.dot(np.power(w.T, self.lamb1), np.power(w, self.lamb2))
        self.W = w1 * w2


class PersonalRank(BiGraphBaseModel):
    """
    基于二部图的Personal算法的实现
    此处是继承自BiGraghBaseModel,所以只实现了迭代的算法，没有实现矩阵求逆算法；
    """
    def __init__(self, N, alpha, steps=0):
        super(PersonalRank, self).__init__(N, steps)
        self.alpha = alpha
        self.current_state = None

    def model_init(self, data_train):
        num_row = data_train.shape[0]
        num_col = data_train.shape[1]
        self.W = np.zeros((num_row + num_col, num_row + num_col))
        sum_row = np.sum(data_train, axis=0)
        sum_col = np.sum(data_train, axis=1)
        sum_row = np.where(sum_row > 0, sum_row, 1.0)       #
        sum_col = np.where(sum_col > 0, sum_col, 1.0)
        self.W[num_row:, :num_row] = data_train.T / sum_col     # 每个分给物品的比例
        self.W[:num_row, num_row:] = data_train / sum_row       # 每个每个物品分给用户的比例
        self.state = np.zeros((num_row + num_col, num_row))
        for i in range(num_row):
            self.state[i, i] = data_train.shape[0]

    def predict(self, data_train, evaluation=False):
        if self.W is None:
            self.model_init(data_train)
            self.current_state = self.state
        while self.current_step < self.steps:
            self.current_state = (1.0 - self.alpha) * self.state + self.alpha * np.dot(self.W, self.current_state)
            self.current_step += 1
        print(self.current_state[data_train.shape[0]:, :].T)
        candidate_orders = self.current_state[data_train.shape[0]:, :].T * (1 - data_train)
        topn_orders = np.argsort(candidate_orders, axis=1)[:, :self.n]  # 去掉用户已经有反馈的商品
        rec_matrix = np.zeros(data_train.shape)
        for i in range(data_train.shape[0]):
            rec_matrix[i, topn_orders[i]] = 1
        return rec_matrix
