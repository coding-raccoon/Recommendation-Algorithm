from .movielens import MovieLensForRatingPred
import numpy as np


if __name__ == '__main__':
    dataset = MovieLensForRatingPred(4)
    data_list = dataset.data_iter()
    for train_data, test_data in data_list:
        print(np.sum(train_data)+np.sum(test_data))
        print(train_data.shape, test_data.shape)