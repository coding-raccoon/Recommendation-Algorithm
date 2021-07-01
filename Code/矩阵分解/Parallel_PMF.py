from DataLoader import MovieLensForRatingPred
from scipy import sparse
import numpy as np
from Evaluation import RatingPredMetrics
import json
import math
import multiprocessing
import time
import os


global_User_mat = None
global_Item_mat = None
global_learning_rate = 0
global_lambda_regularizer = 0
global_num_items = 0
global_num_users = 0
global_K = 0


def init_pool(P_shared, Q_shared, lr, regularizer, num_users, num_items, K):
    global global_User_mat
    global global_Item_mat
    global global_learning_rate
    global global_lambda_regularizer
    global global_num_items
    global global_num_users
    global global_K
    global_User_mat = P_shared
    global_Item_mat = Q_shared
    global_learning_rate = lr
    global_lambda_regularizer = regularizer
    global_num_items = num_items
    global_num_users = num_users
    global_K = K


def SGD(data):
    import numpy as np
    user_id, item_id, score = data[0], data[1], data[2]
    P = np.frombuffer(global_User_mat, dtype=np.float32).reshape(global_num_users, global_K)
    Q = np.frombuffer(global_Item_mat, dtype=np.float32).reshape(global_num_items, global_K)
    error = score - np.dot(P[user_id], Q[item_id].T)
    P[user_id] = P[user_id] - global_learning_rate * (global_lambda_regularizer * P[user_id] - error * Q[item_id])
    Q[item_id] = Q[item_id] - global_learning_rate * (global_lambda_regularizer * Q[item_id] - error * P[user_id])
    # loss.append(0.5 * (error ** 2 + lambda_regularizer * (np.sum(P[user_id] ** 2) + np.sum(Q[item_id] ** 2))))


def Parallel_PMF(train_list, val_list, K, lambda_regularizer=0.1, epochs=100):
    train_coo = sparse.coo_matrix((train_list[:, 2], (train_list[:, 0], train_list[:, 1])))
    val_coo = sparse.coo_matrix((val_list[:, 2], (val_list[:, 0], val_list[:, 1])))
    train_matrix = train_coo.toarray()
    val_matrix = val_coo.toarray()
    num_users, num_items = train_matrix.shape[0], train_matrix.shape[1]
    P = multiprocessing.RawArray("f", np.random.normal(0, 0.1, num_users * K))
    Q = multiprocessing.RawArray("f", np.random.normal(0, 0.1, num_items * K))
    # self.P = np.random.normal(0, 0.1, (num_users, self.K))
    # self.Q = np.random.normal(0, 0.1, (num_items, self.K))
    evaluation = RatingPredMetrics()
    metrics = {'loss': [], 'mae': [], 'rmse': []}
    for epoch in range(epochs):
        start = time.time()
        loss = multiprocessing.Manager().list()
        learning_rate = 0.008 / 2 ** int(math.log(epoch / 10 + 1, 2))
        np.random.shuffle(train_list)
        with multiprocessing.Pool(processes=4, initializer=init_pool, initargs=(P, Q, learning_rate, lambda_regularizer, num_users, num_items, K)) as process_pool:
            process_pool.map(SGD, train_list)
        p = np.frombuffer(P, dtype=np.float32).reshape(num_users, K)
        q = np.frombuffer(Q, dtype=np.float32).reshape(num_items, K)
        val_pred = np.dot(p, q.T)
        # mean_loss = np.mean(loss)
        mae = evaluation.mae(val_matrix, val_pred)
        rmse = evaluation.rmse(val_matrix, val_pred)
        metrics['loss'].append(loss)
        metrics['mae'].append(mae)
        metrics['rmse'].append(rmse)
        end = time.time()
        print("    epoch:{}, lr:{}, mae:{:.4f}, rmse:{:.4f}, cost:{}s".
              format(epoch + 1, learning_rate, mae, rmse, end - start))
    return metrics

if __name__ == "__main__":
    metrics = {'loss': [], 'mae': [], 'rmse': []}
    k = 8
    dataset = MovieLensForRatingPred(5)
    for train_list, val_list in dataset.data_iter():
        print("NUM_DIM:{}".format(k))
        current_metrics = Parallel_PMF(train_list, val_list, k)
        metrics['loss'].append(current_metrics['loss'])
        metrics['mae'].append(current_metrics['mae'])
        metrics['rmse'].append(current_metrics['rmse'])
        k *= 2
    with open('parallel_pmf.json', 'w') as outfile:
        json.dump(metrics, outfile)
