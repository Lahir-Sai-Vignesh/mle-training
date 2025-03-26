#!/usr/bin/env python

import argparse
import os
import pickle

import pandas as pd
from sklearn.impute import SimpleImputer


def load_data(input_dir):
    test_path = os.path.join(input_dir, "test.csv")
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop("median_house_value", axis=1, errors="ignore")
    numeric_features = X_test.select_dtypes(include=["number"]).columns
    categorical_features = X_test.select_dtypes(exclude=["number"]).columns
    imputer = SimpleImputer(strategy="median")
    X_test[numeric_features] = imputer.fit_transform(X_test[numeric_features])
    X_test[categorical_features] = X_test[categorical_features].fillna("Unknown")
    X_test = pd.get_dummies(X_test, columns=categorical_features)
    return X_test


def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def save_predictions(predictions, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}_predictions.csv")
    pd.DataFrame(predictions, columns=["predicted_median_house_value"]).to_csv(
        output_path, index=False
    )
    print(f"Predictions saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Score using trained ML models")
    parser.add_argument("--input", type=str, required=True, help="Input dataset folder")
    parser.add_argument(
        "--models", type=str, required=True, help="Folder containing trained models"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output folder for predictions"
    )
    args = parser.parse_args()

    X_test = load_data(args.input)

    for model_name in ["linear_regression", "decision_tree", "random_forest"]:
        model_path = os.path.join(args.models, f"{model_name}.pkl")
        if os.path.exists(model_path):
            model = load_model(model_path)
            predictions = model.predict(X_test)
            save_predictions(predictions, args.output, model_name)
        else:
            print(f"Warning: Model not found - {model_path}")


if __name__ == "__main__":
    main()
