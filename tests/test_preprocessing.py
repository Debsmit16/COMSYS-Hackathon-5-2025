import numpy as np
from src.data.preprocessing import AdversarialPreprocessor

def test_preprocessing_output_shape():
    pre = AdversarialPreprocessor()
    img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    out = pre.preprocess(img)
    assert out.shape == (224, 224, 3)
    assert (out >= 0).all() and (out <= 1).all()
