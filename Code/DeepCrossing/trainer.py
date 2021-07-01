import kerastuner as kt
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, AUC
from model import DeepCrossingModel
from config import config
from dataloader import CretioDataLoader
import os
import datetime


dataloder = CretioDataLoader(config)
train_iter, val_iter = dataloder.data_iter
feature_columns = dataloder.feature_columns


def model_builder(hp, feature_columns=feature_columns):
    #hp_hidden_unit = hp.Choice('hidden_units', values=config['hidden_units'])
    hp_num_factors = hp.Choice('num_factors', values=config['num_factors'])
    hp_res_dropout = hp.Choice('res_dropout', values=config['res_dropout'])
    hp_embedding_regularizer = hp.Choice('embedding_regularizer', values=config['embedding_regularizer'])
    hp_kernel_regularizer = hp.Choice('kernel_regularizer', values=config['kernel_regularizer'])
    hp_lr = hp.Choice('learning_rate', values=config['learning_rate'])
    model = DeepCrossingModel(feature_columns,
                              config['hidden_units'],
                              hp_num_factors,
                              hp_res_dropout,
                              hp_embedding_regularizer,
                              hp_kernel_regularizer)
    model.compile(optimizer=Adam(hp_lr),
                  loss=BinaryCrossentropy(),
                  metrics=[BinaryAccuracy(), AUC()])
    return model


def train():
    tuner = kt.Hyperband(model_builder,
                         objective=kt.Objective('val_auc', direction='max'),
                         max_epochs=5,
                         factor=3,
                         directory='TuningLog',
                         project_name='DeepCrossing')
    tuner.search(train_iter, validation_data=val_iter, epochs=5)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    with open('./best_hyper', 'w') as f:
        f.write("num_factors: {} \n"
                "res_dropout: {} \nembedding_regularizer: {} \nkernel_regularizer: {} \nlearning_rate: {}"
                .format(best_hps.get('num_factors'),
                        best_hps.get('res_dropout'), best_hps.get('embedding_regularizer'),
                        best_hps.get('kernel_regularizer'), best_hps.get('learning_rate')))
    print("----------------------------------------------")
    print("find best hyperparams as list:")
    print("num_factors: {}".format(best_hps.get("num_factors")))
    print("res_dropout: {}".format(best_hps.get('res_dropout')))
    print("embedding_regularizer: {}".format(best_hps.get('embedding_regularizer')))
    print("kernel_regularizer: {}".format(best_hps.get('kernel_regularizer')))
    print("learning_rate: {}".format(best_hps.get('learning_rate')))
    model = tuner.hypermodel.build(best_hps)
    log = os.path.join("log", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    call_back_1 = ModelCheckpoint('./CheckPoint/DeepCrossing/', monitor='val_auc', verbose=2, save_best_only=True,
                                  save_weights_only=True, mode='max', period=1)
    #call_back_2 = EarlyStopping(monitor='val_rmse', patience=10, verbose=2, mode='min', restore_best_weights=True)
    call_back_2 = TensorBoard(log)
    model.fit(train_iter,
              epochs=20,
              validation_data=val_iter,
              verbose=2,
              callbacks=[call_back_1, call_back_2])