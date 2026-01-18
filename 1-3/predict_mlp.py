import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mlp import SimpleMLP


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="1-3/pre_processed_dataset/dataset/test_processed_v2.csv")
    parser.add_argument("--model", default="1-3/models/mlp_v2.pt")
    parser.add_argument("--out", default="1-3/predictions_v2.csv")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    df = pd.read_csv(args.test)
    target_col = "Transported"
    features = df.drop(columns=[target_col]) if target_col in df.columns else df
    x = features.values.astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP(
        input_dim=x.shape[1],
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        dropout=args.dropout,
    ).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(x).to(device))
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    preds = (probs >= 0.5).astype(int)

    out_df = pd.DataFrame(
        {
            "row_id": np.arange(len(df)),
            "pred": preds,
            "prob": probs,
        }
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"saved predictions to {out_path}")


if __name__ == "__main__":
    main()
