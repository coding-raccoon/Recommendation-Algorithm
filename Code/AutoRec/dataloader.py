import numpy as np
from scipy import sparse
from config import config


class MovieLens(object):
    def __init__(self, DATAPATH=config['DATA_PATH']):
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

    def split_data(self, train_raito=config['train_ratio'], seed=1):
        """
        划分数据，并且返回
        :param train_raito: 训练集所占的比例
        :param seed:随机种子
        :return:返回划分后的训练集矩阵和验证集矩阵
        """
        data_list = self._dataload()
        indices = np.arange(data_list.shape[0])
        np.random.seed(seed)
        np.random.shuffle(indices)
        train_list = data_list[0: int(data_list.shape[0] * train_raito)]
        test_list = data_list[int(data_list.shape[0] * train_raito): ]
        train_coo = sparse.coo_matrix((train_list[:, 2], (train_list[:, 0] - 1, train_list[:, 1] - 1)))
        test_coo = sparse.coo_matrix((test_list[:, 2], (test_list[:, 0] - 1, test_list[:, 1] - 1)))
        return train_coo.toarray(), test_coo.toarray()

    def load_data(self, mode=config['mode']):
        """
        加载数据集
        :param mode: 可选，‘U_AutoRec’和‘I_AutoRec'
        :return: [训练集数组，训练集mask，测试集数组，测试集mask]
        """
        train_matrix, test_matrix = self.split_data()
        train_matrix, test_matrix = train_matrix.astype(np.float32), test_matrix.astype(np.float32)
        train_mask, test_mask = np.where(train_matrix > 0, 1, 0), np.where(test_matrix > 0, 1, 0)
        train_mask, test_mask = train_mask.astype(np.float32), test_mask.astype(np.float32)
        if mode == 'I_AutoRec':
            train_matrix, test_matrix = train_matrix.T, test_matrix.T
            train_mask, test_mask = train_mask.T, test_mask.T
        return train_matrix, train_mask, test_matrix, test_mask


if __name__ == "__main__":
    dataloader = MovieLens()
    for arr in dataloader.load_data():
        print(arr.shape)