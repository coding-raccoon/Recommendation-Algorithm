from dataloader import CretioDataLoder
from model import WideAndDeep
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy, AUC
from tensorflow.keras.losses import BinaryCrossentropy

if __name__ == "__main__":
    dataloader = CretioDataLoder()
    train_iter, val_iter = dataloader.data_iter
    feature_columns = dataloader.feature_columns
    wd_model = WideAndDeep(feature_columns,
                           [256, 128, 64],
                           8,
                           dnn_dropout=0.5)
    wd_model.compile(optimizer=Adam(learning_rate=0.001),
                     loss=BinaryCrossentropy(),
                     metrics=[AUC(), BinaryAccuracy()])
    wd_model.fit(train_iter,
                 epochs=10,
                 validation_data=val_iter,
                 verbose=1)