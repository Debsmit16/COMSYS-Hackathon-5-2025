import numpy as np
from src.models.siamese_network import build_embedding

def test_siamese_embedding_output_shape():
    model = build_embedding("resnet50", (224,224,3), 128)
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    output = model(dummy_input)
    assert output.shape == (1, 128)
    norm = np.linalg.norm(output.numpy())
    assert np.isclose(norm, 1.0, atol=1e-3)
