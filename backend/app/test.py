import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from app.model import FeedForwardNet

def test_loop(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model-path', type=str, default='checkpoints/model.pt')
    args = parser.parse_args()

    device = torch.device(args.device)
    X_test = torch.randn(200, 4)
    y_test = torch.randint(0, 2, (200,))
    loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size)

    model = FeedForwardNet(4, 64, 2).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    test_loop(model, loader, device)

if __name__ == "__main__":
    main()
