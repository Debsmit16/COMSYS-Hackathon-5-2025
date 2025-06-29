import tensorflow as tf
from tensorflow.keras import layers, models, applications
from .triplet_loss import TripletLoss

def build_embedding(backbone="resnet50", input_shape=(224,224,3), embed_dim=128, dropout=0.5):
    if backbone == "resnet50":
        base = applications.ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    elif backbone == "efficientnet_b3":
        base = applications.EfficientNetB3(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Unknown backbone")
    for l in base.layers[:-20]:
        l.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(embed_dim)(x)
    return models.Model(base.input, tf.nn.l2_normalize(x, axis=1), name="embedding")

def build_siamese(cfg):
    emb = build_embedding(cfg["backbone"], embed_dim=cfg["embedding_dim"])
    in_a = layers.Input(cfg["input_shape"], name="anchor")
    in_p = layers.Input(cfg["input_shape"], name="positive")
    in_n = layers.Input(cfg["input_shape"], name="negative")
    out = layers.Concatenate(axis=1)([emb(in_a), emb(in_p), emb(in_n)])
    model = models.Model([in_a, in_p, in_n], out, name="siamese")
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["lr"]), loss=TripletLoss(cfg["triplet_margin"]))
    return model, emb