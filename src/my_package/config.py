import os

# URLs and Paths
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Stratified Split Config
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Feature Engineering
INCOME_BINS = [0.0, 1.5, 3.0, 4.5, 6.0, float("inf")]
INCOME_LABELS = [1, 2, 3, 4, 5]

# Imputation Strategy
IMPUTATION_STRATEGY = "median"

# Hyperparameter Search Space
RANDOM_SEARCH_PARAMS = {
    "n_estimators": (1, 200),
    "max_features": (1, 8),
}
GRID_SEARCH_PARAMS = [
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]
