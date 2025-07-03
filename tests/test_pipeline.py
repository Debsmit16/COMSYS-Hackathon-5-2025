import numpy as np
from src.models.gender_classifier import GenderClassifier

def test_gender_classifier_training_step():
    model = GenderClassifier()
    dummy_x = np.random.rand(2, 224, 224, 3).astype(np.float32)
    dummy_y = np.array([0, 1])
    model.model.compile(optimizer="adam", loss="binary_crossentropy")
    history = model.model.fit(dummy_x, dummy_y, epochs=1, batch_size=2, verbose=0)
    assert history.history["loss"][-1] > 0
