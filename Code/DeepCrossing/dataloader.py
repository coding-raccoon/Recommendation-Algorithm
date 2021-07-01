#
# dataloader.py
#
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf


class CretioDataLoader(object):
    def __init__(self, config):
        self._config = config
        self._load_data()
        self._preprocess()

    def _load_data(self):
        self._original_data = pd.read_csv(self._config['DATAPATH'])

    def _preprocess(self):
        """
        对原始加载的数据做预处理，主要包括以下几个方面：
        - 1.对缺失的数据补全，离散数据补 -1，连续特征补 0；
        - 2.对离散数据重新编号，对连续特征归一化到 [0,1] 之间
        """
        self._sparse_features = [col for col in self._original_data.columns if col[0] == 'C']
        self._dense_features = [col for col in self._original_data.columns if col[0] == 'I']
        # 填充缺失值
        self._original_data[self._sparse_features] = self._original_data[self._sparse_features].fillna('-1')
        self._original_data[self._dense_features] = self._original_data[self._dense_features].fillna('0')
        # 对离散特征进行编码
        for feature in self._sparse_features:
            le = LabelEncoder()
            self._original_data[feature] = le.fit_transform(self._original_data[feature])
        # 将连续特征全部处理到 [0, 1] 以内
        mms = MinMaxScaler()
        self._original_data[self._dense_features] = mms.fit_transform(self._original_data[self._dense_features])
        self._train, self._validation = train_test_split(self._original_data,
                                                         test_size=self._config['validation_split'])

    def _denseFeature(self, feat):
        return {'feat': feat}

    def _sparseFeature(self, feat):
        return {'feat': feat, 'feat_num': len(self._original_data[feat].unique())}

    @property
    def feature_columns(self):
        """
        返回各个特征域的字典，用于模型构建时传入
        """
        return [[self._denseFeature(feat) for feat in self._dense_features],
                [self._sparseFeature(feat) for feat in self._sparse_features]]

    @property
    def data_iter(self):
        train_x = tf.data.Dataset.from_tensor_slices((self._train[self._dense_features].values.astype(np.float32),
                                                      self._train[self._sparse_features].values.astype(np.int32)))
        train_db = tf.data.Dataset.zip((train_x,
                                        tf.data.Dataset.from_tensor_slices(
                                            self._train['label'].values.astype(np.int32))))
        val_x = tf.data.Dataset.from_tensor_slices((self._validation[self._dense_features].values.astype(np.float32),
                                                    self._validation[self._sparse_features].values.astype(np.int32)))
        val_db = tf.data.Dataset.zip((val_x,
                                      tf.data.Dataset.from_tensor_slices(
                                          self._validation['label'].values.astype(np.int32))))
        train_iter = train_db.shuffle(self._config['batch_size'] * 10).batch(self._config['batch_size'])
        val_iter = val_db.shuffle(self._config['batch_size'] * 10).batch(self._config['batch_size'])
        return train_iter, val_iter

    @property
    def load_data(self):
        train_X = [self._train[self._dense_features].values.astype(np.float32),
                   self._train[self._sparse_features].values.astype(np.int32)]
        train_y = self._train['label'].values.astype(np.int32)
        validation_X = [self._validation[self._dense_features].values.astype(np.float32),
                        self._validation[self._sparse_features].values.astype(np.int32)]
        validation_y = self._validation['label'].values.astype(np.int32)
        return train_X, train_y, validation_X, validation_y