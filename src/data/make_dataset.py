import pandas as pd

def clean(df):
    df["quality"] = df["quality"] - 3
    return df

if __name__ == "__main__":
    # read the training data
    train = pd.read_csv('data/raw/train/train.csv').drop('Id',axis=1)

    # combine with original training set
    orig_train = pd.read_csv('data/raw/train/orig_train.csv')
    orig_train = orig_train[~orig_train.duplicated()]
    train = pd.concat([train, orig_train]).reset_index(drop=True)

    # read the test data
    test = pd.read_csv('data/raw/test/test.csv').drop('Id',axis=1)

    train = clean(train)

    train.to_csv('data/processed/train/train.csv', index=False)
    test.to_csv('data/processed/test/test.csv', index=False)