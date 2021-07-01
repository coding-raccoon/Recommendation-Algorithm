import tensorflow as tf
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def data_iter(features, labels, batch_size):
    example_nums=len(features)
    indices=list(range(example_nums))
    np.random.shuffle(indices)
    for i in range(0, example_nums, batch_size):
        index=indices[i:min(i+batch_size, example_nums)]
        yield tf.gather(features,index), tf.gather(labels, index)


class CrossLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, K, **kwargs):
        self.input_dim = input_dim
        self.K = K
        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.V = self.add_weight(name='kernel',
                        shape=(self.input_dim, self.K),
                        initializer='uniform',
                        trainable=True)
        super(CrossLayer, self).build(input_shape)

    @tf.function(input_signature=[tf.TensorSpec(shape = [None,30], dtype = tf.float32)])
    def call(self, x):
        a = tf.reduce_sum(tf.pow(tf.matmul(x, self.V), 2), 1, keepdims=True)
        b = tf.reduce_sum(tf.matmul(tf.pow(x, 2), tf.pow(self.V, 2)), 1, keepdims=True)
        return 0.5 * (a - b)


class FMModel(tf.keras.models.Model):

    def __init__(self, input_dims, K, **kwargs):
        self.input_dims = input_dims
        self.K = K
        super(FMModel, self).__init__(**kwargs)

    def build(self, input_shape):
        self.cross_layer = CrossLayer(self.input_dims, self.K)
        self.linear = tf.keras.layers.Dense(1, bias_regularizer=tf.keras.regularizers.l2(0.01),
                                            kernel_regularizer=tf.keras.regularizers.l1(0.002))
        super(FMModel, self).build(input_shape)

    def call(self, x):
        c = self.cross_layer(x)
        r = self.linear(x)
        return tf.nn.sigmoid(r + c)


optimizer = tf.keras.optimizers.Adam()
loss_func = tf.keras.losses.BinaryCrossentropy()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

valid_loss=tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy=tf.keras.metrics.BinaryAccuracy(name='valid_accuracy')


@tf.function
def printbar():
    today_ts = tf.timestamp() % (24 * 60 * 60)
    # 分别计算出时，分，秒
    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
    minute = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    # 可见尽可能在@function内部使用tf函数
    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return tf.strings.format("0{}", m)
        else:
            return tf.strings.format("{}", m)

    timestring = tf.strings.join([timeformat(hour), timeformat(minute), timeformat(second)], separator=':')
    tf.print("==========" * 8 + timestring)


@tf.function
def train_one_step(model, features, labels):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        predictions = model(features)
        loss = loss_func(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss.update_state(loss)
    train_accuracy.update_state(labels, predictions)


@tf.function
def valid_step(model, features, labels):
    predictions = model(features)
    batch_loss = loss_func(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_accuracy.update_state(labels, predictions)


def train_model(model, train_features, train_labels, val_features, val_labels, epochs):
    for epoch in range(1, epochs + 1):
        for features, labels in data_iter(train_features, train_labels, batch_size=20):
            train_one_step(model, features, labels)

        for features, labels in data_iter(val_features, val_labels, batch_size=20):
            valid_step(model, features, labels)

        if epoch % 10 == 0:
            printbar()
            tf.print(tf.strings.format("Epoch:{},Loss:{},Accuracy:{},Valid loss:{},Valid Accuracy:{}",
                                       [epoch, train_loss.result(), train_accuracy.result(), valid_loss.result(),
                                        valid_accuracy.result()]))

        train_loss.reset_states()
        valid_loss.reset_states()
        train_accuracy.reset_states()
        valid_accuracy.reset_states()


if __name__ == "__main__":
    data = load_breast_cancer()
    features, labels = data.data, np.expand_dims(data.target, axis=1)
    train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size=0.2,
                                                                              random_state=11, stratify=labels)
    model = FMModel(30, 5)
    model.build(input_shape=(None, 30))
    train_model(model, train_features, train_labels, val_features, val_labels, 100)