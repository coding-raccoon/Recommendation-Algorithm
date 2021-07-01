# -*- encoding:utf-8 -*-
import numpy as np
from .base import RatingPredBaseModel


class LFM(RatingPredBaseModel):
    def __init__(self, _lambda=0.1, momentum=0.8):
        pass

    def evaluate(self, data_train, data_validation, num_feats):
        """
        评估超参数，隐语义维度的影响
        :param kargs:
        :param kwargs:
        :return:
        """
        pass

    def compute_metrics(self, data_train, data_validation):
        pass

    def predict(self, *kargs, **kwargs):
        """
        根据训练数据，得到预测的评分矩阵，
        :param kargs:
        :param kwargs:
        :return:
        """
        pass



# class BiasSVD(RatingPredBaseModel):
#     pass
#
#
# class SVDPlus(RatingPredBaseModel):
#     pass

