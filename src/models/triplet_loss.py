import tensorflow as tf
from tensorflow.keras.losses import Loss

class TripletLoss(Loss):
    def __init__(self, margin=0.2, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        embeds = tf.reshape(y_pred, (-1, 3, y_pred.shape[-1]))
        a, p, n = embeds[:, 0], embeds[:, 1], embeds[:, 2]
        pos = tf.reduce_sum(tf.square(a - p), axis=1)
        neg = tf.reduce_sum(tf.square(a - n), axis=1)
        return tf.reduce_mean(tf.maximum(pos - neg + self.margin, 0.0))