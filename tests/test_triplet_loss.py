def test_triplet_loss_zero():
    import tensorflow as tf
    from src.models.triplet_loss import TripletLoss
    z = tf.zeros((6,128)); loss = TripletLoss()(None, z)
    assert abs(float(loss)) < 1e-6