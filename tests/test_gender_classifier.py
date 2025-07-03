import numpy as np
from src.models.gender_classifier import GenderClassifier

def test_gender_classifier_output_shape_and_range():
    model = GenderClassifier()
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    output = model.model(dummy_input)
    assert output.shape == (1, 1)
    assert (output.numpy() >= 0).all() and (output.numpy() <= 1).all()
