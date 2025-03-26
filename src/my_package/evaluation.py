import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from my_package.data_ingestion import prepare_features, preprocess_data
from my_package.training import final_model

# Load and prepare test data
_, test_set = preprocess_data()
X_test, y_test = prepare_features(test_set)


# Evaluate model
def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, predictions)
    return {"RMSE": rmse, "MAE": mae}


# Get final model performance
results = evaluate_model(final_model, X_test, y_test)
print("Final Model Performance:", results)
