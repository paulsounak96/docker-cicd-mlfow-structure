import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from app.model import FeedForwardNet
import mlflow
import mlflow.pytorch
import os

# ------------------------------
# MLflow setup
# ------------------------------
# Use the MLflow container in Docker
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("FeedforwardNN")

# Directory to save local checkpoints
CHECKPOINT_DIR = "/app/app/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ------------------------------
# Training loop
# ------------------------------
def train_loop(model, dataloader, optimizer, criterion, device):
    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

# ------------------------------
# Main function
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Dummy data: 1000 samples, 4 features
    X = torch.randn(1000, 4)
    y = torch.randint(0, 2, (1000,))
    loader = DataLoader(TensorDataset(X, y), batch_size=args.batch_size, shuffle=True)

    model = FeedForwardNet(4, args.hidden_dim, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # ------------------------------
    # MLflow run
    # ------------------------------
    with mlflow.start_run():
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("hidden_dim", args.hidden_dim)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("lr", args.lr)

        for epoch in range(args.epochs):
            train_loop(model, loader, optimizer, criterion, device)
            print(f"Epoch {epoch+1}/{args.epochs} done.")

        # Save locally in checkpoints folder
        local_model_path = os.path.join(CHECKPOINT_DIR, "model.pt")
        torch.save(model.state_dict(), local_model_path)

        # Log model to MLflow server
        mlflow.pytorch.log_model(model, artifact_path="model")

    print(f"✅ Model saved to {local_model_path} and logged to MLflow")

if __name__ == "__main__":
    main()
