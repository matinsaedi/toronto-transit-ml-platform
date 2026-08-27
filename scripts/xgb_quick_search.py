from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


DATA_PATH = "data/raw/TTC Bus Delay Data since 2025.csv"
TARGET = "Min Delay"
BASE_CATEGORICAL_FEATURES = ["Day", "Line", "Station", "Code", "Bound"]
NUMERIC_FEATURES = ["month", "hour"]


@dataclass(frozen=True)
class Experiment:
    name: str
    categorical_features: list[str]
    params: dict


def load_model_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    model_df = df[
        ["Date", "Time", "Day", "Line", "Station", "Code", "Bound", TARGET]
    ].copy()
    model_df["Date"] = pd.to_datetime(model_df["Date"])
    model_df["month"] = model_df["Date"].dt.month
    model_df["hour"] = model_df["Time"].str.split(":").str[0].astype(int)
    model_df["Line"] = model_df["Line"].fillna("UNKNOWN")
    model_df["Bound"] = model_df["Bound"].fillna("UNKNOWN")
    model_df = model_df.drop(columns=["Date", "Time"])
    return model_df


def chronological_split(model_df: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    X = model_df.drop(columns=[TARGET])
    y = model_df[TARGET]

    raw_df = pd.read_csv(DATA_PATH, usecols=["Date", "Time"])
    timestamp = pd.to_datetime(raw_df["Date"].str[:10] + " " + raw_df["Time"])
    sorted_idx = timestamp.sort_values().index

    n = len(sorted_idx)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    train_idx = sorted_idx[:train_end]
    val_idx = sorted_idx[train_end:val_end]
    test_idx = sorted_idx[val_end:]

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_val = X.loc[val_idx]
    y_val = y.loc[val_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test


def run_experiment(
    experiment: Experiment,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                experiment.categorical_features,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )

    t0 = time.perf_counter()
    X_train_t = preprocessor.fit_transform(X_train)
    transform_fit_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    X_val_t = preprocessor.transform(X_val)
    transform_val_seconds = time.perf_counter() - t1

    model = XGBRegressor(**experiment.params)

    t2 = time.perf_counter()
    model.fit(X_train_t, y_train)
    fit_seconds = time.perf_counter() - t2

    t3 = time.perf_counter()
    predictions = model.predict(X_val_t)
    predict_seconds = time.perf_counter() - t3

    mae = mean_absolute_error(y_val, predictions)

    ohe = preprocessor.named_transformers_["categorical"]
    categorical_width = sum(len(categories) for categories in ohe.categories_)

    return {
        "name": experiment.name,
        "features": ",".join(experiment.categorical_features),
        "n_features_after_ohe": categorical_width + len(NUMERIC_FEATURES),
        "val_mae": mae,
        "transform_fit_s": transform_fit_seconds,
        "transform_val_s": transform_val_seconds,
        "fit_s": fit_seconds,
        "predict_s": predict_seconds,
        "params": experiment.params,
    }


def main() -> None:
    model_df = load_model_frame()
    X_train, y_train, X_val, y_val, _, _ = chronological_split(model_df)

    common = {
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    experiments = [
        Experiment(
            name="baseline_exact",
            categorical_features=BASE_CATEGORICAL_FEATURES,
            params={
                "n_estimators": 400,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                **common,
            },
        ),
        Experiment(
            name="shallower_regularized",
            categorical_features=BASE_CATEGORICAL_FEATURES,
            params={
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_child_weight": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                **common,
            },
        ),
        Experiment(
            name="slow_lr_more_trees",
            categorical_features=BASE_CATEGORICAL_FEATURES,
            params={
                "n_estimators": 600,
                "max_depth": 5,
                "learning_rate": 0.03,
                "min_child_weight": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 2.0,
                **common,
            },
        ),
        Experiment(
            name="medium_depth_higher_lr",
            categorical_features=BASE_CATEGORICAL_FEATURES,
            params={
                "n_estimators": 500,
                "max_depth": 4,
                "learning_rate": 0.07,
                "min_child_weight": 5,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "reg_lambda": 2.0,
                **common,
            },
        ),
        Experiment(
            name="depth6_stronger_reg",
            categorical_features=BASE_CATEGORICAL_FEATURES,
            params={
                "n_estimators": 400,
                "max_depth": 6,
                "learning_rate": 0.05,
                "min_child_weight": 10,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "reg_lambda": 5.0,
                **common,
            },
        ),
        Experiment(
            name="depth5_balanced",
            categorical_features=BASE_CATEGORICAL_FEATURES,
            params={
                "n_estimators": 500,
                "max_depth": 5,
                "learning_rate": 0.05,
                "min_child_weight": 1,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "gamma": 0.1,
                **common,
            },
        ),
        Experiment(
            name="compact_strong_reg",
            categorical_features=BASE_CATEGORICAL_FEATURES,
            params={
                "n_estimators": 350,
                "max_depth": 3,
                "learning_rate": 0.08,
                "min_child_weight": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "reg_alpha": 0.2,
                "reg_lambda": 5.0,
                **common,
            },
        ),
        Experiment(
            name="best_no_station_check",
            categorical_features=["Day", "Line", "Code", "Bound"],
            params={
                "n_estimators": 500,
                "max_depth": 4,
                "learning_rate": 0.07,
                "min_child_weight": 5,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "reg_lambda": 2.0,
                **common,
            },
        ),
    ]

    results = [
        run_experiment(experiment, X_train, y_train, X_val, y_val)
        for experiment in experiments
    ]

    results_df = pd.DataFrame(results).sort_values("val_mae").reset_index(drop=True)
    pd.set_option("display.max_colwidth", 120)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
