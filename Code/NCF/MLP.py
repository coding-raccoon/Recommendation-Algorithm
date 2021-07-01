#
# MLP.py
#
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.regularizers import l2


class MLP(Model):
    def __init__(self, num_users, num_items, num_factors, regularizer_lambda=1e-6, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.user_embedding = Embedding(input_dim=num_users,
                                        output_dim=num_factors,
                                        embeddings_initializer='random_normal',
                                        input_length=1)
        self.item_embedding = Embedding(input_dim=num_items,
                                        output_dim=num_factors,
                                        embeddings_initializer='random_normal',
                                        input_length=1)
        self.mlp1 = Dense(64, activation='relu', kernel_regularizer=l2(regularizer_lambda))
        self.mlp2 = Dense(32, activation='relu', kernel_regularizer=l2(regularizer_lambda))
        self.mlp3 = Dense(16, activation='relu', kernel_regularizer=l2(regularizer_lambda))
        self.mlp4 = Dense(8, activation='relu', kernel_regularizer=l2(regularizer_lambda))
        self.affine_output = Dense(1, activation='sigmoid', kernel_regularizer=l2(regularizer_lambda))

    def calculate_latent(self, user_indices, item_indices):
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)
        embedding = tf.concat((user_embedding, item_embedding), axis=1)
        h1 = self.mlp1(embedding)
        h2 = self.mlp2(h1)
        h3 = self.mlp3(h2)
        h4 = self.mlp4(h3)
        return h4

    def call(self, user_indices, item_indices):
        h = self.calculate_latent(user_indices, item_indices)
        y = self.affine_output(h)
        return y