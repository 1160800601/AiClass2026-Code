from pathlib import Path
import json

import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess1():
    print("\n================================== read data")
    df_train = pd.read_csv("dataset/train.csv")
    df_test = pd.read_csv("dataset/test.csv")
    print(df_train.info())
    print(df_test.info())

    # combine datasets
    print("\n================================== combine data")
    df_feat = pd.concat([df_train, df_test], ignore_index=True)
    df_feat = df_feat.drop(["PassengerId", "Name"], axis=1)
    print(df_feat.info())
    print(df_feat)


def preprocess(val_ratio=0.2, random_state=42):
    print("\n================================== read data")
    df_train = pd.read_csv("dataset/train.csv")
    df_test = pd.read_csv("dataset/test.csv")
    test_passenger_id = df_test["PassengerId"].copy()

    # 1) combine datasets
    print("\n================================== combine data")
    df_feat = pd.concat([df_train, df_test], ignore_index=True)
    df_feat = df_feat.drop(["PassengerId", "Name"], axis=1)

    # 2) split Cabin column
    print("\n================================== split Cabin")
    df_feat[["Deck", "Num", "Side"]] = df_feat["Cabin"].str.split("/", expand=True)
    df_feat = df_feat.drop(["Cabin"], axis=1)

    # 3) split back to train/test
    print("\n================================== split train/test")
    df_train_processed = df_feat.iloc[: len(df_train)].copy()
    df_test_processed = df_feat.iloc[len(df_train) :].copy()
    df_test_processed["PassengerId"] = test_passenger_id.values

    # 4) split train/val
    print("\n================================== split train/val")
    df_train_final, df_val_processed = train_test_split(
        df_train_processed,
        test_size=val_ratio,
        random_state=random_state,
        shuffle=True,
    )
    df_train_final = df_train_final.reset_index(drop=True)
    df_val_processed = df_val_processed.reset_index(drop=True)
    print(f"train size {len(df_train_final)}, val size {len(df_val_processed)}")

    return df_train_final, df_val_processed, df_test_processed


def preprocess2(val_ratio=0.2, random_state=42, save_suffix="_v2"):
    print("\n================================== read basic data")
    base_dir = Path("pre_processed_dataset/dataset")
    df_train = pd.read_csv(base_dir / "train_processed_basic.csv")
    df_val = pd.read_csv(base_dir / "val_processed_basic.csv")
    df_test = pd.read_csv(base_dir / "test_processed_basic.csv")
    test_passenger_id = None
    if "PassengerId" in df_test.columns:
        test_passenger_id = df_test["PassengerId"].copy()
        df_test = df_test.drop(columns=["PassengerId"])

    print("\n================================== combine data")
    df_feat = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # 1) add missing flags
    print("\n================================== add missing flags")
    for col in df_feat.columns:
        if col in ["Transported", "PassengerId"]:
            continue
        df_feat[f"{col}_is_missing"] = df_feat[col].isna().astype(int)

    # 2) one-hot encoding
    print("\n================================== one-hot encoding")
    onehot_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]
    existing_onehot = [c for c in onehot_cols if c in df_feat.columns]
    onehot_maps = {}
    for col in existing_onehot:
        onehot_maps[col] = (
            df_feat[col]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )
    df_feat = pd.get_dummies(df_feat, columns=existing_onehot, dummy_na=False)
    bool_cols = df_feat.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df_feat[bool_cols] = df_feat[bool_cols].astype(int)

    maps_dir = Path("pre_processed_dataset/mappings")
    maps_dir.mkdir(parents=True, exist_ok=True)
    maps_path = maps_dir / "onehot_maps.json"
    maps_path.write_text(json.dumps(onehot_maps, ensure_ascii=True, indent=2), encoding="utf-8")

    # 3) min-max scale Age and Num
    print("\n================================== min-max scale Age/Num")
    if "Num" in df_feat.columns:
        df_feat["Num"] = pd.to_numeric(df_feat["Num"], errors="coerce")
    for col in ["Age", "Num"]:
        if col not in df_feat.columns:
            continue
        col_min = df_feat[col].min(skipna=True)
        col_max = df_feat[col].max(skipna=True)
        if pd.notna(col_min) and pd.notna(col_max) and col_max != col_min:
            df_feat[col] = (df_feat[col] - col_min) / (col_max - col_min)
        else:
            df_feat[col] = 0
        df_feat[col] = df_feat[col].fillna(0)

    # 4) log1p + standardize expenses
    print("\n================================== log1p + standardize expenses")
    expense_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in expense_cols:
        if col not in df_feat.columns:
            continue
        series = pd.to_numeric(df_feat[col], errors="coerce")
        series = series.fillna(0)
        series = np.log1p(series)
        mean = series.mean()
        std = series.std()
        if pd.notna(std) and std != 0:
            series = (series - mean) / std
        else:
            series = 0
        df_feat[col] = pd.Series(series).fillna(0)

    # 5) split back to train/val/test
    print("\n================================== split datasets")
    n_train = len(df_train)
    n_val = len(df_val)
    df_train_processed = df_feat.iloc[:n_train].copy()
    df_val_processed = df_feat.iloc[n_train:n_train + n_val].copy()
    df_test_processed = df_feat.iloc[n_train + n_val:].copy()
    if test_passenger_id is not None:
        df_test_processed["PassengerId"] = test_passenger_id.values

    # 6) move Transported to last column in train/val
    if "Transported" in df_train_processed.columns:
        cols = [c for c in df_train_processed.columns if c != "Transported"] + ["Transported"]
        df_train_processed = df_train_processed[cols]
        df_val_processed = df_val_processed[cols]

    bool_cols = df_feat.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df_feat[bool_cols] = df_feat[bool_cols].astype(int)

    if save_suffix:
        save_data(df_train_processed, df_val_processed, df_test_processed, suffix=save_suffix)

    return df_train_processed, df_val_processed, df_test_processed


def save_data(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    suffix: str = "_basic",
):
    out_dir = Path("pre_processed_dataset/dataset")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"train_processed{suffix}.csv"
    val_path = out_dir / f"val_processed{suffix}.csv"
    test_path = out_dir / f"test_processed{suffix}.csv"
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    print(f"\nsaved {train_path} ({len(df_train)} rows)")
    print(f"saved {val_path} ({len(df_val)} rows)")
    print(f"saved {test_path} ({len(df_test)} rows)")


if __name__ == "__main__":
    train_df, val_df, test_df = preprocess(val_ratio=0.2, random_state=42)
    save_data(train_df, val_df, test_df)
    preprocess2(val_ratio=0.2, random_state=42, save_suffix="_v2")
