#!/usr/bin/env python

import argparse
import os
import pickle

import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor


def load_data(input_dir):
    train_path = os.path.join(input_dir, "train.csv")
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop("median_house_value", axis=1)
    y_train = train_df["median_house_value"].copy()
    numeric_features = X_train.select_dtypes(include=["number"]).columns
    categorical_features = X_train.select_dtypes(exclude=["number"]).columns
    imputer = SimpleImputer(strategy="median")
    X_train[numeric_features] = imputer.fit_transform(X_train[numeric_features])
    X_train[categorical_features] = X_train[categorical_features].fillna("Unknown")
    X_train = pd.get_dummies(X_train, columns=categorical_features)
    return X_train, y_train


def train_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_decision_tree(X, y):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)
    return model


def train_random_forest(X, y, cv=5):
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }
    model = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        model,
        param_distributions=param_distribs,
        n_iter=10,
        cv=cv,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(X, y)
    return rnd_search.best_estimator_


def save_model(model, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument("--input", type=str, required=True, help="Input dataset folder")
    parser.add_argument(
        "--output", type=str, required=True, help="Output folder for model pickles"
    )
    args = parser.parse_args()

    X_train, y_train = load_data(args.input)

    models = {
        "linear_regression": train_linear_regression(X_train, y_train),
        "decision_tree": train_decision_tree(X_train, y_train),
        "random_forest": train_random_forest(X_train, y_train),
    }

    for name, model in models.items():
        save_model(model, args.output, name)


if __name__ == "__main__":
    main()
