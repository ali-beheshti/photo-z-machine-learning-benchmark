"""Model registry for photo-z regression."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


def build_model(name: str, random_state: int = 42):
    name = name.lower()

    if name == "rf":
        return RandomForestRegressor(
            n_estimators=20,
            max_depth=20,
            n_jobs=-1,
            random_state=random_state,
        )

    if name == "knn_uniform":
        return KNeighborsRegressor(n_neighbors=20, weights="uniform", n_jobs=-1)

    if name == "knn_distance":
        return KNeighborsRegressor(n_neighbors=30, weights="distance", n_jobs=-1)

    if name == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("Install xgboost to use model='xgb'") from exc
        return XGBRegressor(
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=random_state,
        )

    if name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError("Install catboost to use model='catboost'") from exc
        return CatBoostRegressor(
            iterations=40,
            loss_function="RMSE",
            verbose=False,
            random_seed=random_state,
        )

    if name == "mlp":
        return MLPRegressor(
            max_iter=50,
            random_state=2,
            early_stopping=True,
        )

    if name == "gpr":
        kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3)
        return GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=random_state,
        )

    if name == "keras":
        return _build_keras_model()

    raise KeyError(
        f"Unknown model {name!r}. Choose from "
        "rf, knn_uniform, knn_distance, xgb, catboost, mlp, gpr, keras"
    )


def _build_keras_model():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is optional. Install requirements-optional.txt "
            "to use model='keras'."
        ) from exc

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),  # replaced after preprocessing
    ])
    return model


def build_keras_for_input(n_features: int):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is optional. Install requirements-optional.txt."
        ) from exc

    reg = tf.keras.regularizers.l2()  # coefficient not specified in the project notes
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(60, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(60, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model
