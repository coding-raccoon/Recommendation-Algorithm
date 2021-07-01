import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


class Linear(Layer):
    """
    Wide 部分，就是一个简单的全连接
    """

    def __init__(self, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.dense = Dense(1, activation=None)

    def call(self, x):
        result = self.dense(x)
        return result


class DNN(Layer):
    """
    DNN 层是多个全连接层的堆叠
    输入是经过 Embedding 之后的类别型特征和稠密特征的融合
    """

    def __init__(self, hidden_units, activation='relu', dropout=0., **kwargs):
        super(DNN, self).__init__(**kwargs)
        self.dnn_network = [Dense(unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dropout)

    def call(self, inputs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x


class WideAndDeep(Model):

    def __init__(self,
                 feature_columns,
                 hidden_units,
                 embed_dims,
                 activation='relu',
                 dnn_dropout=0.,
                 embed_reg=1e-4, **kwargs):
        super(WideAndDeep, self).__init__(**kwargs)
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embeding_layers = [Embedding(input_dim=feat['feat_num'],
                                          input_length=1,
                                          output_dim=embed_dims,
                                          embeddings_initializer="random_uniform",
                                          embeddings_regularizer=l2(embed_reg))
                                for i, feat in enumerate(self.sparse_feature_columns)]
        self.deep_network = DNN(hidden_units, activation, dnn_dropout)
        self.wide_network = Linear()
        self.final_dnn = Dense(1, activation=None)

    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs
        sparse_embed = tf.concat([self.embeding_layers[i](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=-1)
        x = tf.concat([sparse_embed, dense_inputs], axis=-1)
        wide_out = self.wide_network(dense_inputs)
        deep_out = self.deep_network(x)
        deep_out = self.final_dnn(deep_out)
        outputs = tf.nn.sigmoid(0.5 * wide_out + 0.5 * deep_out)
        return outputs

    def summary(self):
        dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        Model(inputs=[dense_inputs, sparse_inputs],
              outputs=self.call([dense_inputs, sparse_inputs])).summary()
