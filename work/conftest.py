"""Shared fixtures for P05 tests."""

from pathlib import Path

import pandas as pd
import pytest
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def config():
    with open(HERE / "config.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="session")
def split(config):
    data_path = HERE.parent / "data" / "delivery_times.csv"
    orders = pd.read_csv(data_path)
    X = orders[config["data"]["features"]]
    y = orders[config["data"]["target"]]
    return train_test_split(
        X, y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["seed"],
    )


@pytest.fixture(scope="session")
def trained_model(split):
    X_train, _, y_train, _ = split
    return LinearRegression().fit(X_train, y_train)


@pytest.fixture(scope="session")
def model_mae(trained_model, split):
    _, X_test, _, y_test = split
    return mean_absolute_error(y_test, trained_model.predict(X_test))


@pytest.fixture(scope="session")
def baseline_mae(split):
    _, X_test, y_train, y_test = split
    baseline_predictions = [y_train.mean()] * len(y_test)
    return mean_absolute_error(y_test, baseline_predictions)
