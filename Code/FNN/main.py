from dataloader import CretioDataLoder
from model import FNNModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy, AUC
from tensorflow.keras.losses import BinaryCrossentropy

if __name__ == "__main__":
    dataloader = CretioDataLoder()
    train_iter, val_iter = dataloader.data_iter
    feature_columns = dataloader.feature_columns
    fnn_model = FNNModel(
        feature_columns,
        [128, 128],
        16,
        1e-4,
        1e-4
    )
    fnn_model.compile(
        optimizer=Adam(),
        loss=BinaryCrossentropy(),
        metrics=[BinaryAccuracy(), AUC()]
    )
    fnn_model.fit(
        train_iter,
        epochs=20,
        validation_data=val_iter,
        verbose=1
    )