import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Mean
import numpy as np
import pandas as pd


optimizer = Adam()
loss_func = BinaryCrossentropy()
train_loss = Mean()


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
def train_one_step(model, user_indices, item_indices, rating):
    rating = tf.expand_dims(rating, axis=1)
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        y_pred = model(user_indices, item_indices)
        loss = loss_func(rating, y_pred) + 0.5 * tf.reduce_sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss.update_state(loss)


def evaluate(model, evaluate_data):
    test_users, test_items = evaluate_data[0], evaluate_data[1]
    negative_users, negative_items = evaluate_data[2], evaluate_data[3]
    test_scores = np.squeeze(model(test_users, test_items), axis=1)
    negative_scores = np.squeeze(model(negative_users, negative_items), axis=1)
    test = pd.DataFrame({'user': test_users, 'test_item': test_items, 'test_score': test_scores})
    full = pd.DataFrame({'user': np.concatenate((test_users, negative_users)),
                         'item': np.concatenate((test_items, negative_items)),
                         'score': np.concatenate((test_scores, negative_scores))})
    full = pd.merge(full, test, on=['user'], how='left')
    full['rank'] = full.groupby('user')['score'].rank(method='first', ascending=False)
    full.sort_values(['user', 'rank'], inplace=True)
    top_k = full[full['rank'] <= 10]
    test_in_top_k = top_k[top_k['item'] == top_k['test_item']]
    HR = len(test_in_top_k) * 1.0 / 6040
    test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(lambda x: np.log(2) / np.log(1 + x))
    nDCG = np.sum(test_in_top_k['ndcg']) * 1.0 / 6040
    return HR, nDCG


def train(model, epochs, train_iter, evaluate_data):
    for epoch in range(epochs):
        for step, (user_indices, item_indices, rating) in enumerate(train_iter):
            train_one_step(model, user_indices, item_indices, rating)
        #         evaluate(model, evaluate)
        # if epoch%5 == 0:
        HR, nDCG = evaluate(model, evaluate_data)
        printbar()
        tf.print(tf.strings.format("Epoch: {}, Loss: {}, HR: {}, nDCG: {}", [epoch, train_loss.result(), HR, nDCG]))
        train_loss.reset_states()