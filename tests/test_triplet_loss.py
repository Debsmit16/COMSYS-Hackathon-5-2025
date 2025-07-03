import tensorflow as tf
from src.models.triplet_loss import TripletLoss

def test_triplet_loss_zero():
    margin = 0.2
    z = tf.zeros((6, 128))
    loss = TripletLoss(margin)(None, z)
    assert abs(float(loss) - margin) < 1e-6
