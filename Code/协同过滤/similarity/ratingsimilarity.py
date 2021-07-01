import numpy as np


class RatingPredSimilarity(object):
    def __init__(self):
        super(RatingPredSimilarity, self).__init__()

    def cosine_sim(self, data_train):
        """
        余弦相似度
        :param data_train: 二维数组，数据矩阵，如果计算用户相似度，为user-item矩阵 ；
                                如果计算项目之间相似度，为item-user矩阵
        :return: 相似度矩阵，元素为对应行或列序号的user（或item）之间的相似度
        """
        dotproduct = np.dot(data_train, data_train.T)  # 向量之间的内积
        modulus = np.sqrt(np.sum(np.power(data_train, 2), axis=1))  # 各个行向量的模
        modulus_product = np.dot(modulus.reshape(modulus.shape[0], 1), modulus.reshape(1, modulus.shape[0]))  # 向量模之间的乘积
        modulus_product = np.where(modulus_product == 0, 0.1, modulus_product)  # 避免出现除0的情况
        sim_matrix = dotproduct / modulus_product
        for i in range(sim_matrix.shape[0]):
            sim_matrix[i, i] = 0
        return sim_matrix

    def pearson_correlation(self, data_train):
        """
        皮尔逊相关系数
        :param data_train: 二维数组，数据矩阵，如果计算用户相似度，为user-item矩阵 ；
                                如果计算项目之间相似度，为item-user矩阵
        :return: 各个用户评分之间的皮尔逊相关系数或各个物品的得分之间的皮尔逊相关系数
        """
        sum_score = np.sum(data_train, axis=1)
        binary_data = np.where(data_train != 0, 1.0, 0)
        nums_score = np.sum(binary_data, axis=1)
        mean_score = sum_score / np.where(nums_score>0, nums_score, 0.1)
        zero_center_data = (data_train.T - mean_score).T * binary_data
        square_zero_center_data = np.power(zero_center_data, 2)
        union_square_erroe_sum = np.sqrt(np.dot(square_zero_center_data, binary_data.T))
        normalized_modulu_product = union_square_erroe_sum.T * union_square_erroe_sum
        normalized_modulu_product = np.where(normalized_modulu_product != 0, normalized_modulu_product, 0.01)
        sim_matrix = np.dot(zero_center_data, zero_center_data.T) / normalized_modulu_product
        for i in range(sim_matrix.shape[0]):
            sim_matrix[i, i] = 0
        return sim_matrix

    def adjust_cosine_sim(self, data_train):
        """
        修正的余弦相似度只有在基于项目的协同过滤使用，需要减去用户评分的平均值
        :param data_train: 二维数组，item-user 矩阵
        :return: 返回项目之间的修正的余弦相似度
        """
        sum_score = np.sum(data_train, axis=0)
        binary_data = np.where(data_train != 0, 1.0, 0)
        nums_score = np.sum(binary_data, axis=0)
        mean_score = sum_score / np.where(nums_score > 0, nums_score, 0.1)
        zero_center_data = (data_train - mean_score) * binary_data
        square_zero_center_data = np.power(zero_center_data, 2)
        union_square_erroe_sum = np.sqrt(np.dot(square_zero_center_data, binary_data.T))
        normalized_modulu_product = union_square_erroe_sum.T * union_square_erroe_sum
        normalized_modulu_product = np.where(normalized_modulu_product != 0, normalized_modulu_product, 0.01)
        sim_matrix = np.dot(zero_center_data, zero_center_data.T) / normalized_modulu_product
        for i in range(sim_matrix.shape[0]):
            sim_matrix[i, i] = 0
        return sim_matrix