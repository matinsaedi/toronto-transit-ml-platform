import pandas as pd


FEATURE_COLUMNS = [
    "Day",
    "Line",
    "Code",
    "Bound",
    "month",
    "hour",
]

TARGET_COLUMN = "Min Delay"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["Date"] = pd.to_datetime(data["Date"])
    data["month"] = data["Date"].dt.month

    data["hour"] = data["Time"].str.split(":").str[0].astype(int)

    data["Line"] = data["Line"].fillna("UNKNOWN")
    data["Bound"] = data["Bound"].fillna("UNKNOWN")

    return data[FEATURE_COLUMNS + [TARGET_COLUMN]]
