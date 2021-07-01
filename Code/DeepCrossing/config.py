# config = {
#     'DATAPATH': '../Datasets/Cretio/criteo_sampled_data.csv',
#     'validation_split': 0.1,
#     'batch_size': 4096,
#     'hidden_units': [[512, 256, 128, 64, 32], [256, 512, 128, 32], [256, 128, 64]],
#     'num_factors': [4, 8, 16, 32],
#     'res_dropout': [0.1, 0.2, 0.4, 0.8],
#     'embedding_regularizer': [1e-1, 1e-2, 1e-4, 1e-6],
#     'kernel_regularizer': [1e-1, 1e-2, 1e-4],
#     'learning_rate': [1e-2, 1e-3, 1e-4]
# }

config = {
    'DATAPATH': '../Datasets/Cretio/criteo_sampled_data.csv',
    'validation_split': 0.1,
    'batch_size': 4096,
    'hidden_units': [256, 512, 128, 32],
    'num_factors': [8, 16],
    'res_dropout': [0.4, 0.8],
    'embedding_regularizer': [1e-2, 1e-4],
    'kernel_regularizer': [1e-2, 1e-4],
    'learning_rate': [1e-3, 1e-4]
}