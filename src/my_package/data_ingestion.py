import os
import tarfile

import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

# Import constants from config.py
from my_package.config import (
    HOUSING_PATH,
    HOUSING_URL,
    IMPUTATION_STRATEGY,
    INCOME_BINS,
    INCOME_LABELS,
    RANDOM_STATE,
    TEST_SIZE,
)


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def preprocess_data():
    fetch_housing_data()
    housing = load_housing_data()

    housing["income_cat"] = pd.cut(
        housing["median_income"], bins=INCOME_BINS, labels=INCOME_LABELS
    )

    split = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set


def prepare_features(housing):
    housing_num = housing.drop(
        ["ocean_proximity", "median_house_value"], axis=1, errors="ignore"
    )
    imputer = SimpleImputer(strategy=IMPUTATION_STRATEGY)
    X = imputer.fit_transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    return housing_prepared, housing["median_house_value"]
