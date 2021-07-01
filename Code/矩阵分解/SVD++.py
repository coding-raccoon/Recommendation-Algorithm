from DataLoader import MovieLensForRatingPred
from scipy import sparse
import numpy as np
from Evaluation import RatingPredMetrics
import json
import math


class SVDPlus(object):
    def __init__(self, K, lambda_regularizer=0.1, learning_rate=0.002):
        self.K = K
        self.lambda_regularizer = lambda_regularizer
        self.learning_rate = learning_rate
        self.P = None
        self.Q = None
        self.Y = None
        self.BU = None
        self.BI = None
        self.u = 0

    def fit(self, train_list, val_list, epochs=100):
        train_coo = sparse.coo_matrix((train_list[:, 2], (train_list[:, 0], train_list[:, 1])))
        val_coo = sparse.coo_matrix((val_list[:, 2], (val_list[:, 0], val_list[:, 1])))
        train_matrix = train_coo.toarray()
        val_matrix = val_coo.toarray()
        num_users, num_items = train_matrix.shape[0], train_matrix.shape[1]
        self.P = np.random.normal(0, 0.1, (num_users, self.K))
        self.Q = np.random.normal(0, 0.1, (num_items, self.K))
        self.Y = np.random.normal(0, 0.1, (num_items, self.K))
        self.BU = np.zeros([num_users])
        self.BI = np.zeros([num_items])
        self.u = np.mean(train_list[:, 2])
        evaluation = RatingPredMetrics()
        metrics = {'loss': [], 'mae': [], 'rmse': []}
        Ru = np.sum(np.where(train_matrix > 0, 1, 0), axis=1)
        Y_sum = np.zeros((num_users, self.K))
        Ru_index = []
        for i in range(num_users):
            Ru_index.append(np.where(train_matrix[i] > 0))
        for epoch in range(epochs):
            loss = 0.0
            if epoch < 10:
                self.learning_rate = 0.008
            else:
                self.learning_rate = 0.008 / 2 ** (int(math.log(epoch / 10, 2))+1)
            np.random.shuffle(train_list)
            for data in train_list:
                user_id, item_id, score = data[0], data[1], data[2]
                y_sum = np.sum(self.Y[Ru_index[user_id]], axis=0) * Ru[user_id] ** (-0.5)
                error = score - (self.BU[user_id] + self.BI[item_id] + self.u +
                                 np.dot((self.P[user_id] + y_sum), self.Q[item_id].T))
                self.P[user_id] -= self.learning_rate * (self.lambda_regularizer * self.P[user_id] -
                                                         error * self.Q[item_id])
                self.Q[item_id] -= self.learning_rate * (self.lambda_regularizer * self.Q[item_id] -
                                                         error * (self.P[user_id] + y_sum))
                self.BU[user_id] -= self.learning_rate * (self.lambda_regularizer * self.BU[user_id] - error)
                self.BI[item_id] -= self.learning_rate * (self.lambda_regularizer * self.BI[item_id] - error)
                for j in Ru_index[user_id]:
                    self.Y[j] -= self.learning_rate * (self.lambda_regularizer * self.Y[j] -
                                                       error * Ru[user_id] ** (-0.5) * self.Q[item_id])
                    loss += 0.5 * self.lambda_regularizer * np.sum(self.Y[j]**2)
                loss += 0.5 * (error**2 + self.lambda_regularizer *
                               (np.sum(self.P[user_id]**2) + np.sum(self.Q[item_id]**2) + self.BU[user_id]**2 + self.BI[item_id]**2))
            for j in range(num_users):
                Y_sum[j] = np.sum(self.Y[Ru_index[j]], axis=0) * Ru[j] ** (-0.5)
            val_pred = np.dot(self.P + Y_sum, self.Q.T) + self.u + self.BU.reshape(num_users, 1) + self.BI
            loss = loss / len(train_list)
            mae = evaluation.mae(val_matrix, val_pred)
            rmse = evaluation.rmse(val_matrix, val_pred)
            metrics['loss'].append(loss)
            metrics['mae'].append(mae)
            metrics['rmse'].append(rmse)
            print("    epoch:{}, lr:{}, loss:{:.4f}, mae:{:.4f}, rmse:{:.4f}".format(epoch+1, self.learning_rate, loss, mae, rmse))
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
        model = SVDPlus(k)
        current_metrics = model.fit(train_list, val_list)
        metrics['loss'].append(current_metrics['loss'])
        metrics['mae'].append(current_metrics['mae'])
        metrics['rmse'].append(current_metrics['rmse'])
        k *= 2
    with open('SVDPlus.json', 'w') as outfile:
        json.dump(metrics, outfile)