import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from my_package.data_ingestion import prepare_features, preprocess_data
from my_package.training import (
    grid_search_rf,
    train_decision_tree,
    train_linear_regression,
    train_random_forest,
)


@pytest.fixture
def sample_data():
    """Fixture to load and prepare training data."""
    train_set, _ = preprocess_data()
    X_train, y_train = prepare_features(train_set)
    return X_train, y_train


def test_train_linear_regression(sample_data):
    """Test if Linear Regression model is trained properly."""
    X_train, y_train = sample_data
    model = train_linear_regression(X_train, y_train)

    assert isinstance(model, LinearRegression)
    assert hasattr(model, "coef_")  # Model should have coefficients after training


def test_train_decision_tree(sample_data):
    """Test if Decision Tree model is trained properly."""
    X_train, y_train = sample_data
    model = train_decision_tree(X_train, y_train)

    assert isinstance(model, DecisionTreeRegressor)
    assert hasattr(
        model, "feature_importances_"
    )  # Model should have feature importances


def test_train_random_forest(sample_data):
    """Test if Random Forest is trained and returns a valid model."""
    X_train, y_train = sample_data
    model = train_random_forest(X_train, y_train)

    assert isinstance(model, RandomForestRegressor)
    assert hasattr(model, "feature_importances_")  # Should have feature importances


def test_grid_search_rf(sample_data):
    """Test if Grid Search finds the best Random Forest model."""
    X_train, y_train = sample_data
    model = grid_search_rf(X_train, y_train)

    assert isinstance(model, RandomForestRegressor)
    assert hasattr(model, "feature_importances_")  # Should have feature importances
