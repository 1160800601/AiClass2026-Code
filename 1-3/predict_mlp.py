from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mlp import SimpleMLP


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    df_test = pd.read_csv(base_dir / "pre_processed_dataset/dataset/test_processed_v2.csv")
    passenger_id = df_test["PassengerId"].values if "PassengerId" in df_test.columns else None

    drop_cols = [c for c in ["Transported", "PassengerId"] if c in df_test.columns]
    df_test_features = df_test.drop(drop_cols, axis=1)
    df_test_features = df_test_features.apply(pd.to_numeric, errors="coerce").fillna(0)

    X_test = torch.tensor(df_test_features.values, dtype=torch.float32)
    model = SimpleMLP()
    model_path = base_dir / "models/mlp_acc0.7964_ep24.pt"
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    with torch.no_grad():
        probs = model(X_test).squeeze(1).numpy()
    preds = (probs >= 0.5)

    if passenger_id is not None:
        out_df = pd.DataFrame({"PassengerId": passenger_id, "Transported": preds})
    else:
        out_df = pd.DataFrame({"row_id": np.arange(len(df_test)), "Transported": preds})

    out_path = base_dir / "submission.csv"
    out_df.to_csv(out_path, index=False)
    print(f"saved predictions to {out_path}")


if __name__ == "__main__":
    main()
