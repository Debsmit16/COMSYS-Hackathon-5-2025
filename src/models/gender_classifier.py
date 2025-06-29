import numpy as np, tensorflow as tf
from tensorflow.keras import layers, models, applications

class GenderClassifier:
    def __init__(self, backbone="efficientnet_b3", input_shape=(224,224,3), dropout=0.3):
        if backbone=="efficientnet_b3":
            base = applications.EfficientNetB3(weights="imagenet", include_top=False, input_shape=input_shape)
        else:
            base = applications.ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
        for l in base.layers[:-15]:
            l.trainable = False
        x = layers.GlobalAveragePooling2D()(base.output)
        x = layers.Dense(512, activation="relu")(x); x = layers.BatchNormalization()(x); x = layers.Dropout(dropout)(x)
        x = layers.Dense(256, activation="relu"); x = layers.BatchNormalization()(x); x = layers.Dropout(dropout)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        self.model = models.Model(base.input, out)
        self.model.compile(optimizer="adam", loss="binary_crossentropy",
                           metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    def predict_gender(self, img):
        p = float(self.model(np.expand_dims(img,0))[0][0])
        return ("Male", p) if p>0.5 else ("Female", 1-p)