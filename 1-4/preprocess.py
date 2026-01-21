import pandas as pd


def preprocess(train_path, test_path):
    # 读入 CSV 数据
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    print(df_train.info())
    print(df_test.info())
    
    # 处理训练集数据
    # TODO: 开始你的表演，请不要直接复制代码！
    
    
    return train_data, train_label, test_data
