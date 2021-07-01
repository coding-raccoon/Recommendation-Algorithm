import random
import pandas as pd
import numpy as np
from copy import deepcopy
import tensorflow as tf


class DataGenrator(object):
    def __init__(self, config):
        self._config = config
        self._loadData()
        self._preprocess()
        self._binarize(self._originalRatings)
        self._userPool = set(self._originalRatings['userId'].unique())
        self._itemPool = set(self._originalRatings['itemId'].unique())
        self._select_Negatives(self._preprocessRatings)
        self._split_pool(self._preprocessRatings)

    def _loadData(self):
        """
        加载原始的数据为 DataFrame
        """
        self._originalRatings = pd.read_csv(self._config['DATAPATH'],
                                            sep='::',
                                            header=None,
                                            names=['uid', 'mid', 'rating', 'timestamp'],
                                            engine='python')

        return self._originalRatings

    def _preprocess(self):
        """
        对原始的数据进行处理，重新编号
        """
        # 新建名为 "userId" 的列，这列对用户从 0 开始编号
        user_id = self._originalRatings[['uid']].drop_duplicates().reset_index()
        user_id['userId'] = np.arange(len(user_id))
        self._originalRatings = pd.merge(self._originalRatings, user_id, on=['uid'], how='left')
        # 新建名为 "itemId" 的列，这列对物品编号
        item_id = self._originalRatings[['mid']].drop_duplicates().reset_index()
        item_id['itemId'] = np.arange(len(item_id))
        self._originalRatings = pd.merge(self._originalRatings, item_id, on=['mid'], how='left')
        # 按照 ['userId', 'itemId', 'rating', 'timestamp'] 的顺序重新排列
        self._originalRatings = self._originalRatings[['userId', 'itemId', 'rating', 'timestamp']]

    def _binarize(self, ratings):
        """
        对显示反馈全部置为 1
        """
        ratings = deepcopy(ratings)
        ratings['rating'][ratings['rating'] > 0] = 1.0
        self._preprocessRatings = ratings

    def _select_Negatives(self, ratings):
        """
        构造每一个用户的负反馈集合，并且每个用户负采样 99 个负样本，作为评估的时候使用
        """
        # 构造每一个用户的正反馈集合
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        # 构造每一个用户的负反馈集合
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self._itemPool - x)
        # 从负反馈集合中随机采样 99 个样本在评估的时候使用
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 99))
        self._negatives = interact_status[['userId', 'negative_items', 'negative_samples']]

    def _split_pool(self, ratings):
        """
        leave one out train/test split
        """
        # 先将数据按照 "userId" 进行分组，然后按照时间戳降序排列
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        # 根据论文 leave one out train/test split 的策略，应该选择（最近的反馈）时间戳最大的作为测试
        test = ratings[ratings['rank_latest'] == 1]
        train = ratings[ratings['rank_latest'] > 1]
        self.train_ratings = train[['userId', 'itemId', 'rating']]
        self.test_ratings = test[['userId', 'itemId', 'rating']]

    def _sample_generator(self):
        """
        对每一个正反馈，就进行一定次数的负采样，最终生成整个训练数据集
        """
        train_ratings = pd.merge(self.train_ratings, self._negatives[['userId', 'negative_items']], on='userId')
        train_ratings['negatives'] = train_ratings['negative_items'].apply(
            lambda x: random.sample(x, self._config['num_negative']))
        users, items, ratings = [], [], []
        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            for i in range(self._config['num_negative']):
                users.append(int(row.userId))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))
        return users, items, ratings

    @property
    def train_iter(self):
        users, items, ratings = self._sample_generator()
        users = np.array(users)
        items = np.array(items)
        ratings = np.array(ratings)
        train_db = tf.data.Dataset.from_tensor_slices((users, items, ratings))
        train_iter = train_db.shuffle(self._config['batch_size'] * 10).batch(self._config['batch_size'])
        return train_iter

    @property
    def evaluate_data(self):
        test_ratings = pd.merge(self.test_ratings, self._negatives[['userId', 'negative_samples']], on='userId')
        test_users, test_items, negative_users, negative_items = [], [], [], []
        for row in test_ratings.itertuples():
            test_users.append(row.userId)
            test_items.append(row.itemId)
            for i in range(len(row.negative_samples)):
                negative_users.append(row.userId)
                negative_items.append(row.negative_samples[i])
        return [np.array(test_users), np.array(test_items), np.array(negative_users), np.array(negative_items)]
