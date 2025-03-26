import pytest

from my_package.data_ingestion import prepare_features, preprocess_data
from my_package.evaluation import evaluate_model
from my_package.training import final_model


@pytest.fixture
def sample_test_data():
    """Fixture to load and prepare test data."""
    _, test_set = preprocess_data()
    X_test, y_test = prepare_features(test_set)
    return X_test, y_test


def test_evaluate_model(sample_test_data):
    """Test if evaluate_model runs without errors and returns expected metrics."""
    X_test, y_test = sample_test_data
    results = evaluate_model(final_model, X_test, y_test)

    assert isinstance(results, dict)
    assert "RMSE" in results and "MAE" in results
    assert results["RMSE"] >= 0, "RMSE should not be negative"
    assert results["MAE"] >= 0, "MAE should not be negative"


def test_model_prediction_shape(sample_test_data):
    """Ensure model predictions have the correct shape."""
    X_test, y_test = sample_test_data
    predictions = final_model.predict(X_test)

    assert (
        predictions.shape == y_test.shape
    ), "Predictions should match test labels in shape"
