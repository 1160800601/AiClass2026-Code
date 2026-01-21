import pandas as pd

def split_data():
    df = pd.read_csv("dataset/train.csv")
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    df.iloc[:split_idx].to_csv("dataset/train1.csv", index=False)
    df.iloc[split_idx:].to_csv("dataset/val1.csv", index=False)
    return

def preprocess(train_path, test_path):
    # 读入 CSV 数据
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    print(df_train.info())
    print(df_test.info())
    # Process training data.
    train_label = df_train.iloc[:, 0].to_numpy(copy=False)
    train_data = df_train.iloc[:, 1:].to_numpy(copy=False)
    train_data = train_data.reshape(-1, 1, 28, 28)
    # Process test data.
    test_data = df_test.to_numpy(copy=False)
    test_data = test_data.reshape(-1, 1, 28, 28)
    # Normalize pixel values.
    train_data = train_data / 255.0
    test_data = test_data / 255.0
    return train_data, train_label, test_data

# if __name__ == '__main__':
    # split_data()


