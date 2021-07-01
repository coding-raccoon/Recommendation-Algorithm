import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense,Embedding
from tensorflow.keras.regularizers import l2

class FNNModel(Model):

    def __init__(self,
                 feature_columns,
                 hidden_units,
                 num_factors,
                 embedding_regularizer,
                 kernel_regularizer,
                 **kwargs):
        super(FNNModel, self).__init__(**kwargs)
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.linear_embeddings = [Embedding(input_dim=feat['feat_num'],
                                            output_dim=1,
                                            input_length=1,
                                            embeddings_initializer='random_uniform',
                                            embeddings_regularizer=l2(embedding_regularizer))
                                  for i, feat in enumerate(self.sparse_feature_columns)]
        self.crossing_embeddings = [Embedding(input_dim=feat['feat_num'],
                                              output_dim=num_factors,
                                              input_length=1,
                                              embeddings_initializer='random_uniform',
                                              embeddings_regularizer=l2(embedding_regularizer))
                                    for i, feat in enumerate(self.sparse_feature_columns)]
        self.dnn_layers = [Dense(unit, activation='tanh') for unit in hidden_units]
        self.dnn = Dense(1, activation='sigmoid')

    def call(self, x):
        dense_input, sparse_input = x
        linear_embedding = tf.concat([self.linear_embeddings[i](sparse_input[:, i])
                                      for i in range(sparse_input.shape[1])], axis=-1)
        sparse_embedding = tf.concat([self.crossing_embeddings[i](sparse_input[:, i])
                                      for i in range(sparse_input.shape[1])], axis=-1)
        stacking = tf.concat([linear_embedding, sparse_embedding, dense_input], axis=-1)
        for layer in self.dnn_layers:
            stacking = layer(stacking)
        y = self.dnn(stacking)
        return y

    def summary(self):
        dense_input = Input(shape=(len(self.dense_feature_columns),))
        sparse_input = Input(shape=(len(self.sparse_feature_columns),))
        Model(inputs=[dense_input, sparse_input], outputs=self.call([dense_input, sparse_input])).summary()