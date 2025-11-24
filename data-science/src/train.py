# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
import os
import logging
from pathlib import Path
import pandas as pd
import joblib  # For fallback model save
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature  # For model signature
from azureml.core import Run

# Set MLflow logging to DEBUG for better error visibility in Azure logs
logging.getLogger("mlflow").setLevel(logging.DEBUG)

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument("--mlflow_log_model", type=str, help="Path to write MLflow model folder")
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='The number of trees in the forest')
    parser.add_argument('--max_depth', type=int, default=5,
                        help='The maximum depth of the tree')
    parser.add_argument('--criterion', type=str, default='squared_error',
                    help='The function to measure the quality of a split')

    args = parser.parse_args()

    return args

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # Ensure model output dir exists
    os.makedirs(args.model_output, exist_ok=True)
    print(f"[Diagnostic] Model output path: {args.model_output}")

    # Read train and test data (assuming CSV files in the provided paths; adjust if dir-based)
    # Note: In Azure ML, paths are mounted dirs, so check if they contain the CSVs
    train_file = Path(args.train_data) / "train.csv"
    test_file = Path(args.test_data) / "test.csv"
    
    if not train_file.exists():
        raise FileNotFoundError(f"Train CSV not found at {train_file}. Check mounted input path: {args.train_data}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test CSV not found at {test_file}. Check mounted input path: {args.test_data}")
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    print(f"Loaded {len(train_df)} train rows and {len(test_df)} test rows.")

    # Split the data into features(X) and target(y)
    y_train = train_df['Price']
    X_train = train_df.drop(columns=['Price'])
    y_test = test_df['Price']
    X_test = test_df.drop(columns=['Price'])

    # Log input shapes for debugging
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))
    mlflow.log_param("num_features", X_train.shape[1])

    # Initialize and train a RandomForest Regressor
    model = RandomForestRegressor(
        n_estimators=args.n_estimators, 
        max_depth=args.max_depth, 
        criterion=args.criterion, 
        random_state=42
    )
    model.fit(X_train, y_train)
    print(f"Model trained with n_estimators={args.n_estimators}, max_depth={args.max_depth}, criterion={args.criterion}")

    # Predict on test data
    yhat_test = model.predict(X_test)

    # Compute and log MSE
    mse = mean_squared_error(y_test, yhat_test)
    print(f'Mean Squared Error of RandomForest Regressor on test set: {mse:.2f}')
    mlflow.log_metric("MSE", float(mse))

    # Log hyperparameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("criterion", args.criterion)

    # Infer signature and prepare input example
    predictions = model.predict(X_test)
    signature = infer_signature(X_test, predictions)
    input_example = X_test.iloc[:5].to_dict('records')  # Small sample for logging

    # Log model with enhanced options and error handling
    try:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            conda_env=None  # Use pip reqs from env; avoids old sklearn issues
        )
        # Save model to pipeline output folder
        print("Model logged successfully via MLflow!")

        # Save model to pipeline output folder
        mlflow.sklearn.save_model(model, args.mlflow_log_model)
        print(f"MLflow model saved to pipeline output folder: {args.mlflow_log_model}")
    
    except Exception as e:
        print(f"MLflow logging failed: {e}")
        # Fallback: Save model manually and log as artifact
        fallback_path = os.path.join(args.model_output, "model.pkl")
        joblib.dump(model, fallback_path)
        mlflow.log_artifact(fallback_path, artifact_path=args.model_output)
        print(f"Fallback model saved to {fallback_path} and logged as artifact.")
        raise  # Re-raise to flag the issue in Azure logs, but artifacts still exist

if __name__ == "__main__":
    # Parse Arguments
    args = parse_args()
    
    # Diagnostic prints
    print(f"[Diagnostic] Train dataset input path: {args.train_data}")
    print(f"[Diagnostic] Test dataset input path: {args.test_data}")
    print(f"[Diagnostic] Model output path argument: {args.model_output}")
    print(f"Number of Estimators: {args.n_estimators}")
    print(f"Max Depth: {args.max_depth}")

    # Start MLflow run explicitly (ensures active run for all logging)
    run = Run.get_context()
    with mlflow.start_run(run_id=run.id):
        main(args)
        # No explicit end_run needed; context manager handles it
