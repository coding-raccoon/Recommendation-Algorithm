from GMF import GMF
from MLP import MLP
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Dense
from tensorflow.keras.regularizers import l2


class NeuMF(Model):
    def __init__(self, num_users, num_items, num_factors, gmf_regularizer=0, mlp_regularizer=1e-6, **kwargs):
        super(NeuMF, self).__init__(**kwargs)
        self.gmf = GMF(num_users, num_items, num_factors, gmf_regularizer, **kwargs)
        self.mlp = MLP(num_users, num_items, num_factors, mlp_regularizer, **kwargs)
        self.affine_output = Dense(1, activation='sigmoid')

    def call(self, user_indices, item_indices):
        gmf_vector = self.gmf.calculate_latent(user_indices, item_indices)
        mlp_vector = self.mlp.calculate_latent(user_indices, item_indices)
        h = tf.concat((gmf_vector, mlp_vector), axis=1)
        return self.affine_output(h)