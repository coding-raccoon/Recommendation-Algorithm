import numpy as np
# import dataset


class TopNSimilarity(object):
    """
    用于计算TopN推荐需要的user（或item) 之间的相似度
    """
    def __int__(self):
        super(TopNSimilarity, self).__init__()

    def jaccard_sim(self, matrix):
        """
        计算jaccard相似度
        :param matrix: 二维数组，数据矩阵，如果计算用户相似度，为user-item矩阵 ；
                        如果计算项目之间相似度，为item-user矩阵
        :return: 相似度矩阵，元素为对应行或列序号的user（或item）之间的相似度
        """
        intersection = np.dot(matrix, matrix.T)     # 计算user（或item）两两之间的交集
        activity_user = np.sum(matrix, axis=1)
        cross_sum = activity_user.reshape(len(activity_user), 1) + activity_user.reshape(1, len(activity_user))
        union = cross_sum - intersection
        union = np.where(union == 0, 0.1, union)        # 计算user（或item）两两之间的并集
        sim_matrix = intersection / union
        for i in range(sim_matrix.shape[0]):
            sim_matrix[i, i] = 0
        return sim_matrix

    def cossine_sim(self, matrix):
        """
        余弦相似度
        :param matrix: 二维数组，数据矩阵，如果计算用户相似度，为user-item矩阵 ；
                        如果计算项目之间相似度，为item-user矩阵
        :return: 相似度矩阵，元素为对应行或列序号的user（或item）之间的相似度
        """
        dotproduct = np.dot(matrix, matrix.T)       # 向量之间的内积
        modulus = np.sqrt(np.sum(np.power(matrix, 2), axis=1))      # 各个行向量的模
        modulus_product = np.dot(modulus.reshape(modulus.shape[0], 1), modulus.reshape(1, modulus.shape[0]))        # 向量模之间的乘积
        modulus_product = np.where(modulus_product == 0, 0.1, modulus_product)      # 避免出现除0的情况
        sim_matrix = dotproduct / modulus_product
        for i in range(sim_matrix.shape[0]):
            sim_matrix[i, i] = 0
        return sim_matrix

    def penalty_cossine_sim(self, matrix):
        """
        带惩罚项的余弦相似度，需要考虑用户的活跃度或者是物品的流行度，加上惩罚项
        :param matrix: 二维数组，数据矩阵，如果计算用户相似度，为user-item矩阵 ；
                        如果计算项目之间相似度，为item-user矩阵
        :return: 相似度矩阵，元素为对应行或列序号的user（或item）之间的相似度
        """
        item_weights = np.log(matrix.shape[0] / (np.sum(matrix, axis=0) + 1))       # 计算各行代表的user（或item）的惩罚权重
        weighted_matrix = matrix * item_weights
        weighted_dotproduct = np.dot(weighted_matrix, matrix.T)     # 带惩罚项权重的内积
        modulus = np.sqrt(np.sum(np.power(matrix, 2), axis=1))
        modulus_product = np.dot(modulus.reshape(modulus.shape[0], 1), modulus.reshape(1, modulus.shape[0]))
        modulus_product = np.where(modulus_product == 0, 0.1, modulus_product)
        sim_matrix = weighted_dotproduct / modulus_product
        for i in range(sim_matrix.shape[0]):
            sim_matrix[i, i] = 0
        return sim_matrix

    def johnbreese_sim(self, matrix):
        """
        johnbreese相似度，也可以看作是一种带惩罚项的余弦相似度
        :param matrix: 二维数组，数据矩阵，如果计算用户相似度，为user-item矩阵 ；
                        如果计算项目之间相似度，为item-user矩阵
        :return: 相似度矩阵，元素为对应行或列序号的user（或item）之间的相似度
        """
        item_weights = np.log(1.01 + np.sum(matrix, axis=0))        # 计算各行代表的user（或item）的惩罚权重
        item_weights = 1.0 / item_weights
        weighted_matrix = matrix * np.expand_dims(item_weights, axis=0)
        weighted_dotproduct = np.dot(weighted_matrix, matrix.T)     # 带惩罚项权重的内积
        modulus = np.sqrt(np.sum(np.power(matrix, 2), axis=1))
        modulus_product = np.dot(modulus.reshape(modulus.shape[0], 1), modulus.reshape(1, modulus.shape[0]))
        modulus_product = np.where(modulus_product == 0, 0.1, modulus_product)
        sim_matrix = weighted_dotproduct / modulus_product
        for i in range(sim_matrix.shape[0]):
            sim_matrix[i, i] = 0
        return sim_matrix