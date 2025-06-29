import yaml, tensorflow as tf
from src.data.dataset import make_tf_dataset
from src.models.gender_classifier import GenderClassifier

def main():
    cfg = yaml.safe_load(open("config/config.yaml"))
    train_ds, val_ds = make_tf_dataset(cfg["data"], task="gender")
    model = GenderClassifier(backbone=cfg["model"]["gender"]["backbone"], dropout=cfg["model"]["gender"]["dropout_rate"])
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=15),
        tf.keras.callbacks.ModelCheckpoint("models/final/gender_model.h5", save_best_only=True)
    ]
    model.model.fit(train_ds, validation_data=val_ds, epochs=cfg["training"]["epochs"], callbacks=callbacks)

if __name__=="__main__":
    main()