
config = {
    'mode': 'I_AutoRec',
    'DATA_PATH': '../Datasets/MovieLens/ratings.dat',
    'num_users': 6040,
    'num_items': 3952,
    'train_ratio': 0.9,
    'epochs': 100,
    'batch_size': 64,
    'hidden_units': [32, 64, 128, 256, 512],
    'l2_lambda': [0.01, 0.1, 1.0],
    'activation': ['sigmoid', 'relu', 'tanh'],
    'learning_rate': [0.1, 0.01, 0.001, 0.0001]
}