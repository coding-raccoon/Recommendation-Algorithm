import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf


class CretioDataLoder(object):
    def __init__(self):
        self._datapath = "../Datasets/Cretio/criteo_sampled_data.csv"
        self._test_size = 0.2
        self._batch_size = 4096
        self._load_data()
        self._preprocess()

    # load orginal data
    def _load_data(self):
        self._data = pd.read_csv(self._datapath)

    # data preprocessing
    def _preprocess(self):
        self._data.drop("I12", axis=1, inplace=True)
        self._data.drop("C22", axis=1, inplace=True)
        self._dense_columns = [column for column in self._data.columns.values if column[0] == 'I']
        self._sparse_columns = [column for column in self._data.columns.values if column[0] == 'C']
        for column in self._dense_columns:
            self._data[column] = self._data[column].fillna(self._data[column].mean())
        self._data[self._sparse_columns] = self._data[self._sparse_columns].fillna('0')
        for column in self._sparse_columns:
            le = LabelEncoder()
            self._data[column] = le.fit_transform(self._data[column])
        mms = MinMaxScaler()
        self._data[self._dense_columns] = mms.fit_transform(self._data[self._dense_columns])
        self._train, self._validation = train_test_split(self._data, test_size=self._test_size)

        # dense feature dict

    def _denseFeature(self, feat):
        return {'feat': feat}

    # sparse feature dict
    def _sparseFeature(self, feat):
        return {'feat': feat, 'feat_num': len(self._data[feat].unique())}

    @property
    def feature_columns(self):
        return [[self._denseFeature(feat) for feat in self._dense_columns],
                [self._sparseFeature(feat) for feat in self._sparse_columns]]

    @property
    def data_iter(self):
        train_x = tf.data.Dataset.from_tensor_slices((self._train[self._dense_columns].values.astype(np.float32),
                                                      self._train[self._sparse_columns].values.astype(np.int32)))
        train_db = tf.data.Dataset.zip((train_x,
                                        tf.data.Dataset.from_tensor_slices(
                                            self._train['label'].values.astype(np.int32))))
        val_x = tf.data.Dataset.from_tensor_slices((self._validation[self._dense_columns].values.astype(np.float32),
                                                    self._validation[self._sparse_columns].values.astype(np.int32)))
        val_db = tf.data.Dataset.zip((val_x,
                                      tf.data.Dataset.from_tensor_slices(
                                          self._validation['label'].values.astype(np.int32))))
        train_iter = train_db.shuffle(self._batch_size * 10).batch(self._batch_size)
        val_iter = val_db.shuffle(self._batch_size * 10).batch(self._batch_size)
        return train_iter, val_iter

    @property
    def load_data(self):
        train_X = [self._train[self._dense_columns].values.astype(np.float32),
                   self._train[self._sparse_columns].values.astype(np.int32)]
        train_y = self._train['label'].values.astype(np.int32)
        validation_X = [self._validation[self._dense_columns].values.astype(np.float32),
                        self._validation[self._sparse_columns].values.astype(np.int32)]
        validation_y = self._validation['label'].values.astype(np.int32)
        return train_X, train_y, validation_X, validation_y