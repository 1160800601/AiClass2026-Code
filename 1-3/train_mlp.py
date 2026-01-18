import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from mlp import SimpleMLP


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path: Path, target_col: str) -> tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    x = df.drop(columns=[target_col]).values.astype(np.float32)
    y = df[target_col].values.astype(np.float32).reshape(-1, 1)
    return torch.tensor(x), torch.tensor(y)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="1-3/pre_processed_dataset/dataset/train_processed_v2.csv")
    parser.add_argument("--val", default="1-3/pre_processed_dataset/dataset/val_processed_v2.csv")
    parser.add_argument("--out", default="1-3/models/mlp_v2.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train = load_dataset(Path(args.train), "Transported")
    x_val, y_val = load_dataset(Path(args.val), "Transported")

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)

    model = SimpleMLP(
        input_dim=x_train.shape[1],
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        dropout=args.dropout,
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / max(len(train_loader.dataset), 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch:03d} train_loss {train_loss:.4f} val_loss {val_loss:.4f} val_acc {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_path)

    print(f"saved model to {out_path}")


if __name__ == "__main__":
    main()
