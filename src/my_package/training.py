from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

from my_package.data_ingestion import prepare_features, preprocess_data

# Load and prepare data
train_set, test_set = preprocess_data()
X_train, y_train = prepare_features(train_set)


# Train models
def train_linear_regression(X, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    return lin_reg


def train_decision_tree(X, y):
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(X, y)
    return tree_reg


def train_random_forest(X, y):
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }
    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(X, y)
    return rnd_search.best_estimator_


def grid_search_rf(X, y):
    param_grid = [
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]
    forest_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(X, y)
    return grid_search.best_estimator_


# Train and store models
lin_model = train_linear_regression(X_train, y_train)
tree_model = train_decision_tree(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)
final_model = grid_search_rf(X_train, y_train)
