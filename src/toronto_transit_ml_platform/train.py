from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from toronto_transit_ml_platform.data import load_data, split_data
from toronto_transit_ml_platform.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    prepare_features,
)
from toronto_transit_ml_platform.model_io import save_model


DATA_PATH = Path("data/raw/TTC Bus Delay Data since 2025.csv")
MODEL_PATH = Path("artifacts/models/xgb_pipeline.joblib")

def build_model() -> Pipeline:
    categorical_features = [
        "Day",
        "Line",
        "Code",
        "Bound",
    ]

    numeric_features = [
        "month",
        "hour",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
        ]
    )

    regressor = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("regressor", regressor),
        ]
    )

    return model


def split_xy(model_df):
    X = model_df[FEATURE_COLUMNS]
    y = model_df[TARGET_COLUMN]

    return X, y


def train():
    # 1. Load raw TTC data
    df = load_data(DATA_PATH)

    # 2. Chronological train / validation / test split
    train_df, val_df, test_df = split_data(df)

    # 3. Feature engineering
    train_model_df = prepare_features(train_df)
    val_model_df = prepare_features(val_df)
    test_model_df = prepare_features(test_df)

    # 4. Separate features and target
    X_train, y_train = split_xy(train_model_df)
    X_val, y_val = split_xy(val_model_df)
    X_test, y_test = split_xy(test_model_df)

    # 5. Build and train the full preprocessing + model pipeline
    model = build_model()
    model.fit(X_train, y_train)

    # 6. Evaluate on validation data
    val_predictions = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_predictions)

    print(f"Validation MAE: {val_mae:.4f}")

    save_model(model, MODEL_PATH)

    return model, {
        "validation_mae": val_mae,
    }


if __name__ == "__main__":
    train()
