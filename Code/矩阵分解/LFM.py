from DataLoader import MovieLensForRatingPred
from scipy import sparse
import numpy as np
from Evaluation import RatingPredMetrics
import json
import math
import time


class PMF(object):
    def __init__(self, K, lambda_regularizer=0.1, learning_rate=0.004):
        self.K = K
        self.lambda_regularizer = lambda_regularizer
        self.learning_rate = learning_rate
        self.P = None
        self.Q = None

    def fit(self, train_list, val_list, epochs=100):
        train_coo = sparse.coo_matrix((train_list[:, 2], (train_list[:, 0], train_list[:, 1])))
        val_coo = sparse.coo_matrix((val_list[:, 2], (val_list[:, 0], val_list[:, 1])))
        train_matrix = train_coo.toarray()
        val_matrix = val_coo.toarray()
        num_users, num_items = train_matrix.shape[0], train_matrix.shape[1]
        self.P = np.random.normal(0, 0.1, (num_users, self.K))
        self.Q = np.random.normal(0, 0.1, (num_items, self.K))
        evaluation = RatingPredMetrics()
        metrics = {'loss': [], 'mae': [], 'rmse': []}
        for epoch in range(epochs):
            start = time.time()
            loss = 0.0
            self.learning_rate = 0.008 / 2**int(math.log(epoch/10 + 1, 2))
            np.random.shuffle(train_list)
            for data in train_list:
                user_id, item_id, score = data[0], data[1], data[2]
                error = score - np.dot(self.P[user_id], self.Q[item_id].T)
                self.P[user_id] = self.P[user_id] - self.learning_rate * (self.lambda_regularizer * self.P[user_id] -
                                                                          error * self.Q[item_id])
                self.Q[item_id] = self.Q[item_id] - self.learning_rate * (self.lambda_regularizer * self.Q[item_id] -
                                                                          error * self.P[user_id])
                # loss += 0.5 * (error**2 + self.lambda_regularizer *
                #                (np.sum(self.P[user_id]**2) + np.sum(self.Q[item_id]**2)))
            val_pred = np.dot(self.P, self.Q.T)
            loss = loss / len(train_list)
            mae = evaluation.mae(val_matrix, val_pred)
            rmse = evaluation.rmse(val_matrix, val_pred)
            metrics['loss'].append(loss)
            metrics['mae'].append(mae)
            metrics['rmse'].append(rmse)
            end = time.time()
            print("    epoch:{}, lr:{}, loss:{:.4f}, mae:{:.4f}, rmse:{:.4f}, cost:{}".
                  format(epoch+1, self.learning_rate, loss, mae, rmse, end - start))
        return metrics

    def predoict_one(self, user_id, item_id):
        return np.dot(self.P[user_id], self.Q[item_id])

    def predict_all(self):
        return np.dot(self.P, self.Q)


if __name__ == "__main__":
    metrics = {'loss': [], 'mae': [], 'rmse': []}
    k = 8
    dataset = MovieLensForRatingPred(5)
    for train_list, val_list in dataset.data_iter():
        print("NUM_DIM:{}".format(k))
        model = PMF(k)
        current_metrics = model.fit(train_list, val_list)
        metrics['loss'].append(current_metrics['loss'])
        metrics['mae'].append(current_metrics['mae'])
        metrics['rmse'].append(current_metrics['rmse'])
        k *= 2
    with open('pmf.json', 'w') as outfile:
        json.dump(metrics, outfile)
