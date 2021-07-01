#
# GMF.py
#
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.regularizers import l2


class GMF(Model):

    def __init__(self, num_users, num_items, num_factors, regularizer_lambda=0, **kwargs):
        super(GMF, self).__init__(**kwargs)
        self.user_embedding = Embedding(input_dim=num_users, output_dim=num_factors, input_length=1)
        self.item_embedding = Embedding(input_dim=num_items, output_dim=num_factors, input_length=1)
        self.affine_output = Dense(1, activation='sigmoid', kernel_regularizer=l2(regularizer_lambda))

    def calculate_latent(self, user_indices, item_indices):
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)
        element_product = tf.multiply(user_embedding, item_embedding)
        return element_product

    def call(self, user_indices, item_indices):
        element_product = self.calculate_latent(user_indices, item_indices)
        y = self.affine_output(element_product)
        return y