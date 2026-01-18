import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(val_ratio=0.2, random_state=42):
    print("\n================================== 读入数据")
    df_train = pd.read_csv('1-3/dataset/train.csv')
    df_test = pd.read_csv('1-3/dataset/test.csv')
    print(df_train.info())
    print(df_test.info())
    
    # 1. 合并数据集
    print("\n================================== 合并数据")
    # 合并训练集和测试集
    df_feat = pd.concat([df_train, df_test], ignore_index=True)
    # 删除无关列
    df_feat = df_feat.drop(['PassengerId', 'Name'], axis=1)
    print(df_feat.info())
    print(df_feat)
    
    # 2. 分离 Cabin 列
    print("\n================================== 分离 Cabin 列")
    # 把 Cabin 列分为 Deck、Num、Side 三列
    df_feat[['Deck','Num','Side']] = df_feat['Cabin'].str.split('/', expand=True)
    # 删除 Cabin 列
    df_feat = df_feat.drop(['Cabin'], axis=1)
    print(df_feat.info())

    # 3. 处理数值型数据
    print("\n================================== 处理数值型数据")
    # 提取类型为数值的列
    num_cols = df_feat.columns[df_feat.dtypes != 'object']
    # 归一化（均值为0，方差为1）
    df_feat[num_cols] = df_feat[num_cols].apply(lambda x: (x - x.mean()) / x.std())
    # 缺失值填充为0
    df_feat[num_cols] = df_feat[num_cols].fillna(0)
    print(df_feat.info())
    print(df_feat.describe())
    
    # 4. 处理类别型数据
    print("\n================================== 处理类别型数据")
    # 提取类型为类别的列
    cate_cols = df_feat.columns[df_feat.dtypes == 'object']
    # 用整数编码替换类别，NAN值用-1替换
    df_feat[cate_cols] = df_feat[cate_cols].apply(lambda x: pd.Categorical(x).codes)
    print(df_feat.info())
    print(df_feat)
    
    # 5. 分离训练集和测试集
    print("\n================================== 分离训练集和测试集")
    # 从总数据中，再分出训练集和测试集
    df_train_processed = df_feat.iloc[:len(df_train)].copy()
    df_test_processed = df_feat.iloc[len(df_train):].copy()
    # 重新设置id
    df_train_processed['PassengerId'] = df_train['PassengerId'].values
    df_test_processed['PassengerId'] = df_test['PassengerId'].values
    print(df_train_processed.info())
    print(df_train_processed)
    print(df_test_processed.info())
    print(df_test_processed)
    
    # 6. 划分训练集和验证集
    print("\n================================== 划分训练集和验证集")
    split_result = train_test_split(
        df_train_processed, 
        test_size=val_ratio, 
        random_state=random_state,
        shuffle=True
    )
    df_train_final: pd.DataFrame = split_result[0]
    df_val_processed: pd.DataFrame = split_result[1]
    # 重置索引
    df_train_final = df_train_final.reset_index(drop=True)
    df_val_processed = df_val_processed.reset_index(drop=True)
    print(f"训练集大小: {len(df_train_final)}, 验证集大小: {len(df_val_processed)}")
    print(f"验证集占比: {val_ratio * 100:.1f}%")
    
    return df_train_final, df_val_processed, df_test_processed

def save_data(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    df_train.to_csv('1-3/dataset/train_processed.csv', index=False)
    df_val.to_csv('1-3/dataset/val_processed.csv', index=False)
    df_test.to_csv('1-3/dataset/test_processed.csv', index=False)
    print(f"\n已保存: train_processed.csv ({len(df_train)} 条)")
    print(f"已保存: val_processed.csv ({len(df_val)} 条)")
    print(f"已保存: test_processed.csv ({len(df_test)} 条)")


if __name__ == '__main__':
    train_df, val_df, test_df = preprocess(val_ratio=0.2, random_state=42)
    save_data(train_df, val_df, test_df)
