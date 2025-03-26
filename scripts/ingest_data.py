#!/usr/bin/env python

import argparse
import os
import tarfile
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def fetch_housing_data(housing_url):
    """Download and extract the dataset from the given URL."""
    dataset_name = os.path.splitext(
        os.path.basename(urllib.parse.urlparse(housing_url).path)
    )[0]
    dataset_dir = os.path.join(
        "datasets", dataset_name
    )  # Store inside 'datasets/' for organization
    os.makedirs(dataset_dir, exist_ok=True)

    tgz_path = os.path.join(dataset_dir, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)

    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=dataset_dir)

    return dataset_dir


def load_housing_data(dataset_dir):
    """Load housing data from extracted CSV."""
    csv_path = os.path.join(dataset_dir, "housing.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected file not found: {csv_path}")
    return pd.read_csv(csv_path)


def preprocess_data(dataset_dir, output_dir):
    """Preprocess data and save train/test splits."""
    housing = load_housing_data(dataset_dir)

    # Creating an income category column for stratification
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_idx]
        strat_test_set = housing.loc[test_idx]

    # Dropping the 'income_cat' column after stratification
    for dataset in (strat_train_set, strat_test_set):
        dataset.drop("income_cat", axis=1, inplace=True)

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    strat_train_set.to_csv(train_path, index=False)
    strat_test_set.to_csv(test_path, index=False)

    print(f"Training data saved to {train_path}")
    print(f"Test data saved to {test_path}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare housing dataset")
    parser.add_argument(
        "url",
        type=str,
        help="URL of the dataset to download",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output folder path for storing processed dataset",
    )
    args = parser.parse_args()

    dataset_dir = fetch_housing_data(args.url)
    preprocess_data(dataset_dir, args.output)


if __name__ == "__main__":
    main()
