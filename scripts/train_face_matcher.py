import yaml, tensorflow as tf
from src.data.dataset import make_tf_dataset
from src.models.siamese_network import build_siamese

def main():
    cfg = yaml.safe_load(open("config/config.yaml"))
    train_ds, val_ds = make_tf_dataset(cfg["data"], task="face")
    siam_cfg={
        "backbone":cfg["model"]["siamese"]["backbone"],
        "embedding_dim":cfg["model"]["siamese"]["embedding_dim"],
        "input_shape":tuple(cfg["data"]["image_size"])+(3,),
        "lr":cfg["training"]["learning_rate"],
        "triplet_margin":cfg["model"]["siamese"]["triplet_margin"]
    }
    model, emb = build_siamese(siam_cfg)
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=15),
        tf.keras.callbacks.ModelCheckpoint("models/final/face_embedding.h5", save_best_only=True)
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=cfg["training"]["epochs"], callbacks=callbacks)

if __name__=="__main__":
    main()