import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.regularizers import l2


class AutoRecModel(Model):
    def __init__(self, hidden_units, activations, l2_lambdas, output_dims, **kwargs):
        super(AutoRecModel, self).__init__(**kwargs)
        self.hidden_unit = hidden_units
        self.activation = activations
        self.l2_lambda = l2_lambdas
        self.loss_tracker = Mean(name='loss')
        self.metric_tracker = Mean(name='rmse')
        self.encoder = Dense(self.hidden_unit, activation=self.activation, kernel_regularizer=l2(self.l2_lambda))
        self.decoder = Dense(output_dims, kernel_regularizer=l2(self.l2_lambda))

    def _loss(self, train_data, train_mask, y_pred):
        se_loss = tf.reduce_sum(tf.pow(train_data - y_pred * train_mask, 2))
        l2_loss = self.losses
        return se_loss + l2_loss, se_loss

    @property
    def metrics(self):
        return [self.loss_tracker, self.metric_tracker]

    def train_step(self, data):
        train_data, train_mask = data
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            y_pred = self(train_data, training=True)
            loss, se_loss = self._loss(train_data, train_mask, y_pred)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        rmse = tf.sqrt(se_loss / tf.reduce_sum(train_mask))
        self.loss_tracker.update_state(loss)
        self.metric_tracker.update_state(rmse)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        train_data, val_data, val_mask = data
        y_pred = self(train_data)
        loss, se_loss = self._loss(val_data, val_mask, y_pred)
        rmse = tf.sqrt(se_loss / tf.reduce_sum(val_mask))
        self.loss_tracker.update_state(loss)
        self.metric_tracker.update_state(rmse)
        return {m.name: m.result() for m in self.metrics}

    def call(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out