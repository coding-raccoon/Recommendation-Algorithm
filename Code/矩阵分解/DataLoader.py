# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse


DATAPATH = '../Dataset/MovieLens/ratings.dat'


class MovieLensForRatingPred(object):
    """
    加载movielens数据集用于评分预测
    """
    def __init__(self, M):
        """
        初始化数据集加载算法
        :param M: 数据集交叉验证的折数
        """
        self.M = M
        self.path = DATAPATH

    def _dataload(self):
        """
        读取数据文件，生成[用户id，项目id，评分]的三元组
        :return: 所有三元组组成的列表
        """
        with open(self.path) as f:
            lines = f.readlines()
        data_list = [list(map(int, line.strip().split('::')[:3])) for line in lines]
        np.random.shuffle(data_list)
        return np.array(data_list)

    def data_iter(self, seed=1):
        data_list = self._dataload()
        np.random.seed(seed)
        data_numbering = np.random.randint(0, self.M, data_list.shape[0])
        for i in range(self.M):
            mask = np.where(data_numbering == i, True, False)
            test_list = data_list[mask] - 1     # 矩阵下标从0开始，但是用户和物品id均从1开始
            train_list = data_list[~mask] - 1
            train_list[:, 2] += 1
            test_list[:, 2] += 1
            yield train_list, test_list