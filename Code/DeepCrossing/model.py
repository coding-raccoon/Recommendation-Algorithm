#
# model.py
#
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, ReLU, Dropout, Embedding, Layer
from tensorflow.keras.regularizers import l2


class Residual_Units(Layer):
    """
    残差单元，连接在模型 stacking 层之后
    """

    def __init__(self, hidden_units, input_dims, kernel_regularizer, **kwargs):
        super(Residual_Units, self).__init__(**kwargs)
        self.layer1 = Dense(hidden_units, kernel_regularizer=l2(kernel_regularizer), activation='relu')
        self.layer2 = Dense(input_dims, kernel_regularizer=l2(kernel_regularizer), activation=None)
        self.relu = ReLU()

    def call(self, x):
        h = self.layer1(x)
        y = self.layer2(h)
        output = self.relu(x + y)
        return output


class DeepCrossingModel(Model):
    def __init__(self,
                 featurn_columns,
                 hidden_units,
                 num_factors,
                 res_dropout,
                 embedding_regularizer,
                 kernel_regularizer,
                 **kwargs):
        super(DeepCrossingModel, self).__init__(**kwargs)
        self.dense_feature_columns, self.sparse_feature_columns = featurn_columns
        self.embedding_layers = [Embedding(input_dim=feat['feat_num'],
                                           output_dim=num_factors,
                                           input_length=1,
                                           embeddings_initializer='random_uniform',
                                           embeddings_regularizer=l2(embedding_regularizer))
                                 for i, feat in enumerate(self.sparse_feature_columns)]
        self.stacking_dims = num_factors * len(self.sparse_feature_columns) + len(self.dense_feature_columns)
        self.residual_layers = [Residual_Units(unit, self.stacking_dims, kernel_regularizer) for unit in hidden_units]
        self.droput = Dropout(res_dropout)
        # self.dense1 = Dense(hiddens, activation='relu')
        self.dense2 = Dense(1, activation='sigmoid')

    def call(self, x):
        dense_input, sparse_input = x
        sparse_embedding = tf.concat([self.embedding_layers[i](sparse_input[:, i])
                                      for i in range(sparse_input.shape[1])], axis=-1)
        stacking = tf.concat([sparse_embedding, dense_input], axis=-1)
        for residual_layer in self.residual_layers:
            stacking = residual_layer(stacking)
        # h = self.dense1(stacking)
        y = self.dense2(stacking)
        return y

    def summary(self):
        dense_inputs = Input(shape=(len(self.dense_feature_columns),))
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),))
        Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()