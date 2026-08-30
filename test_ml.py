import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, "data", "census.csv")


@pytest.fixture(scope="module")
def data():
    """Load the census dataset once and share it across the tests."""
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def train_test_data(data):
    """Split the census data into train/test sets used by the tests below."""
    train, test = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data["salary"],
    )
    return train, test


@pytest.fixture(scope="module")
def processed_and_trained(train_test_data):
    """Process the training data and fit a model for the tests."""
    train, test = train_test_data
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    model = train_model(X_train, y_train)
    return model, X_test, y_test


def test_train_model_returns_random_forest(processed_and_trained):
    """
    Test that train_model returns a fitted RandomForestClassifier, i.e. that
    the pipeline uses the algorithm we expect and the model is ready to predict.
    """
    model, _, _ = processed_and_trained
    assert isinstance(model, RandomForestClassifier)
    # A fitted forest exposes the estimators it built during fit().
    assert hasattr(model, "estimators_")
    assert len(model.estimators_) == 100


def test_inference_returns_expected_type_and_shape(processed_and_trained):
    """
    Test that inference returns a numpy array of binary predictions with one
    prediction per row of the input data.
    """
    model, X_test, y_test = processed_and_trained
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]
    assert preds.shape[0] == y_test.shape[0]
    # The task is binary classification, so only 0 and 1 are valid labels.
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_returns_expected_values():
    """
    Test that compute_model_metrics returns the correct precision, recall and
    F1 for a known set of labels and predictions.
    """
    y = np.array([1, 1, 1, 0, 0, 0])
    preds = np.array([1, 1, 0, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    # 2 true positives, 1 false positive, 1 false negative.
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(2 / 3)


def test_train_test_split_sizes_and_columns(data, train_test_data):
    """
    Test that the train/test split produces the expected sizes, keeps every
    column, and that both splits contain each salary class.
    """
    train, test = train_test_data
    assert len(train) + len(test) == len(data)
    assert len(test) == pytest.approx(len(data) * 0.20, abs=1)
    assert list(train.columns) == list(data.columns)
    assert set(train["salary"].unique()) == set(test["salary"].unique())