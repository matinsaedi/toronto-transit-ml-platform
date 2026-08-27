import pandas as pd


def load_data(path):
    return pd.read_csv(path)

def split_data(df):
    data = df.copy()

    data["timestamp"] = pd.to_datetime(
        data["Date"].str[:10] + " " + data["Time"]
    )

    data = data.sort_values("timestamp")

    n = len(data)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    train_df = data.iloc[:train_end].copy()
    val_df = data.iloc[train_end:val_end].copy()
    test_df = data.iloc[val_end:].copy()

    return train_df, val_df, test_df
