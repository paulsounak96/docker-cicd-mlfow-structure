# backend/app/tests/test_train.py
from app.model import FeedForwardNet
import torch

def test_model_forward():
    model = FeedForwardNet(input_dim=10, hidden_dim=32, output_dim=2)
    x = torch.randn(5, 10)
    y = model(x)
    assert y.shape == (5, 2)   # output should match batch_size x output_dim