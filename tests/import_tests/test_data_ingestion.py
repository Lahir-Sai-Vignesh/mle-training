import os

import pandas as pd
import pytest

from my_package.data_ingestion import (
    fetch_housing_data,
    prepare_features,
    preprocess_data,
)

HOUSING_PATH = os.path.join("datasets", "housing")


def test_fetch_housing_data():
    """Test if the data is fetched and extracted properly"""
    fetch_housing_data()
    assert os.path.exists(os.path.join(HOUSING_PATH, "housing.csv"))


def test_preprocess_data():
    """Test if preprocess_data returns train and test datasets"""
    train_set, test_set = preprocess_data()
    assert isinstance(train_set, pd.DataFrame)
    assert isinstance(test_set, pd.DataFrame)
    assert not train_set.empty
    assert not test_set.empty


def test_prepare_features():
    """Test if feature engineering is applied correctly"""
    train_set, _ = preprocess_data()
    prepared_features, labels = prepare_features(train_set)

    assert isinstance(prepared_features, pd.DataFrame)
    assert isinstance(labels, pd.Series)
    assert "rooms_per_household" in prepared_features.columns
    assert "bedrooms_per_room" in prepared_features.columns
    assert "population_per_household" in prepared_features.columns
