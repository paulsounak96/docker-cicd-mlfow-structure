# backend/app/tests/test_train.py
from train import FeedforwardNN
import torch

def test_model_forward():
    model = FeedforwardNN(input_dim=10, hidden_dim=32, output_dim=2)
    x = torch.randn(5, 10)
    y = model(x)
    assert y.shape == (5, 2)   # output should match batch_size x output_dim