"""Preprocessing, model fitting, scoring, and plots."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from .features import split_feature_types
from .metrics import score_photoz
from .models import build_model, build_keras_for_input


def make_preprocessor(columns: list[str]) -> ColumnTransformer:
    numerical, categorical = split_feature_types(columns)

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num_pipe, numerical),
        ("cat", cat_pipe, categorical),
    ])


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model_name: str,
    target: str = "zspec",
    random_state: int = 42,
):
    pre = make_preprocessor(features)
    x_train = pre.fit_transform(train[features])
    x_test = pre.transform(test[features])
    y_train = train[target].to_numpy()
    y_test = test[target].to_numpy()

    if model_name == "keras":
        model = build_keras_for_input(x_train.shape[1])
        model.fit(
            x_train, y_train,
            epochs=40,
            batch_size=200,
            verbose=0,
            validation_split=0.1,
        )
        pred = model.predict(x_test, verbose=0).ravel()
    else:
        model = build_model(model_name, random_state=random_state)
        # Full Gaussian-process regression is cubic in N; cap the demonstration.
        if model_name == "gpr" and len(x_train) > 5000:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(x_train), size=5000, replace=False)
            model.fit(x_train[idx], y_train[idx])
        else:
            model.fit(x_train, y_train)
        pred = model.predict(x_test)

    return pred, score_photoz(y_test, pred)


def save_photoz_plot(y_true, y_pred, output: str | Path, title: str = ""):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dz = (y_pred - y_true) / (1.0 + y_true)
    outlier = np.abs(dz) > 0.15

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true[~outlier], y_pred[~outlier], s=5, alpha=0.35, label="inlier")
    ax.scatter(y_true[outlier], y_pred[outlier], s=7, alpha=0.55, label="outlier")
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1)
    ax.set_xlabel(r"$z_{\rm spec}$")
    ax.set_ylabel(r"$z_{\rm phot}$")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
