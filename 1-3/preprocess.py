from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess1():
    print("\n================================== read data")
    df_train = pd.read_csv("1-3/dataset/train.csv")
    df_test = pd.read_csv("1-3/dataset/test.csv")
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
    df_train = pd.read_csv("1-3/dataset/train.csv")
    df_test = pd.read_csv("1-3/dataset/test.csv")

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


def preprocess2(val_ratio=0.2, random_state=42):
    print("\n================================== read data")
    df_train = pd.read_csv("1-3/dataset/train.csv")
    df_test = pd.read_csv("1-3/dataset/test.csv")
    print(df_train.info())
    print(df_test.info())

    # 1) combine datasets
    print("\n================================== combine data")
    df_feat = pd.concat([df_train, df_test], ignore_index=True)
    df_feat = df_feat.drop(["PassengerId", "Name"], axis=1)
    print(df_feat.info())
    print(df_feat)

    # 2) split Cabin column
    print("\n================================== split Cabin")
    df_feat[["Deck", "Num", "Side"]] = df_feat["Cabin"].str.split("/", expand=True)
    df_feat = df_feat.drop(["Cabin"], axis=1)
    print(df_feat.info())

    # 3) numeric features
    print("\n================================== numeric features")
    num_cols = df_feat.columns[df_feat.dtypes != "object"]
    df_feat[num_cols] = df_feat[num_cols].apply(lambda x: (x - x.mean()) / x.std())
    df_feat[num_cols] = df_feat[num_cols].fillna(0)
    print(df_feat.info())
    print(df_feat.describe())

    # 4) categorical features
    print("\n================================== categorical features")
    cate_cols = df_feat.columns[df_feat.dtypes == "object"]
    df_feat[cate_cols] = df_feat[cate_cols].apply(lambda x: pd.Categorical(x).codes)
    print(df_feat.info())
    print(df_feat)

    # 5) split train/test
    print("\n================================== split train/test")
    df_train_processed = df_feat.iloc[: len(df_train)].copy()
    df_test_processed = df_feat.iloc[len(df_train) :].copy()
    df_train_processed["PassengerId"] = df_train["PassengerId"].values
    df_test_processed["PassengerId"] = df_test["PassengerId"].values
    print(df_train_processed.info())
    print(df_train_processed)
    print(df_test_processed.info())
    print(df_test_processed)

    # 6) split train/val
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
    print(f"val ratio {val_ratio * 100:.1f}%")

    return df_train_final, df_val_processed, df_test_processed


def save_data(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    suffix: str = "_basic",
):
    out_dir = Path("1-3/pre_processed_dataset/dataset")
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
