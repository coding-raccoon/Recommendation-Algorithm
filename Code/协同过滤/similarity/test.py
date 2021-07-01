# if __name__ == "__main__":
#     rating_test_true = np.array([[1.0, 4.0, 0, 5.0], [2.0, 0, 0, 4.0], [1.0, 4.0, 3.0, 5.0]])
#     rating_test_pred = np.array([[3.0, 2.0, 0, 2.0], [5.0, 0, 0, 4.0], [2.0, 4.0, 5.0, 3.0]])
#     Metrics = RatingPredSimilarity()
#     print(Metrics.cosine_sim(rating_test_true.T))
#     print(Metrics.adjust_cosine_sim(rating_test_true.T))
#     print(Metrics.pearson_correlation(rating_test_true.T))
#     # print(Metrics.rmse(rating_test_true, rating_test_pred))