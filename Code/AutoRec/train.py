import tensorflow as tf
import kerastuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from model import AutoRecModel
from config import config
from dataloader import MovieLens
import os
import datetime
# import IPython

BATCH_SIZE = config['batch_size']


def model_builder(hp):
    if config['mode'] == 'I_AutoRec':
        output_dims = config['num_users']
    else:
        output_dims = config['num_items']
    hp_units = hp.Choice('hidden_unit', values=config['hidden_units'])
    hp_activations = hp.Choice('activation', values=config['activation'])
    hp_l2_lambda = hp.Choice('l2_lambda', values=config['l2_lambda'])
    hp_lr = hp.Choice('learning_rate', values=config['learning_rate'])
    model = AutoRecModel(hp_units, hp_activations, hp_l2_lambda, output_dims)
    model.compile(optimizer=Adam(learning_rate=hp_lr))
    return model


def data_preprocessing():
    dataset = MovieLens()
    train_matrix, train_mask, val_matrix, val_mask = dataset.load_data()
    train_db = tf.data.Dataset.from_tensor_slices((train_matrix, train_mask))
    train_iter = train_db.shuffle(BATCH_SIZE * 10).batch(BATCH_SIZE)
    val_db = tf.data.Dataset.from_tensor_slices((train_matrix, val_matrix, val_mask))
    val_iter = val_db.shuffle(BATCH_SIZE * 10).batch(BATCH_SIZE)
    return train_iter, val_iter


# class ClearTrainingOutput(Callback):
#     def on_train_end(*args, **kargs):
#         IPython.display.clear_output(wait=True)


def train():
    tuner = kt.Hyperband(model_builder, objective=kt.Objective('rmse', direction='min'), max_epochs=50, factor=3,
                         directory='TuningLog', project_name='autorec')
    train_iter, val_iter = data_preprocessing()
    tuner.search(train_iter, validation_data=val_iter, epochs=50)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    with open('./best_hyper', 'w') as f:
        f.write("hidden_units:{} \nl2_lambda: {} \nactivation: {} \nlearning_rate: {}"
                .format(best_hps.get('hidden_unit'), best_hps.get('l2_lambda'),
                        best_hps.get('activation'), best_hps.get('learning_rate')))
    print("----------------------------------------------")
    print("find best hyperparams as list:")
    print("hidden_units: {}".format(best_hps.get('hidden_unit')))
    print("l2_lambda: {}".format(best_hps.get("l2_lambda")))
    print("activation: {}".format(best_hps.get('activation')))
    print("learning_rate: {}".format(best_hps.get('learning_rate')))
    model = tuner.hypermodel.build(best_hps)
    log = os.path.join("log", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    call_back_1 = ModelCheckpoint('./CheckPoint/AutoRec/', monitor='val_rmse', verbose=2, save_best_only=True,
                                  save_weights_only=True, mode='min', period=5)
    call_back_2 = EarlyStopping(monitor='val_rmse', patience=10, verbose=2, mode='min', restore_best_weights=True)
    call_back_3 = TensorBoard(log)
    model.fit(train_iter,
              epochs=100,
              validation_data=val_iter,
              verbose=2,
              callbacks=[call_back_1, call_back_2, call_back_3])
