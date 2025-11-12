# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""
import argparse
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test-train ratio")
    args = parser.parse_args()
    return args

def main(args):
    '''Read, preprocess, split, and save datasets'''
    # Auth workaround for student workspaces
    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, subscription_id="be39bd0a-b78a-4b84-9646-d4021e417375", resource_group_name="default_resource_group", workspace_name="weeksix")
    print("Using DefaultAzureCredential for auth")
    # Reading Data
    df = pd.read_csv(args.raw_data)
    # Encode categorical feature
    le = LabelEncoder()
    df['Segment'] = le.fit_transform(df['Segment'])
    # Split Data
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)
    # Save
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)
    train_df.to_csv(os.path.join(args.train_data, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "test.csv"), index=False)
    # Preview
    print("\n✅ Preview of train.csv:")
    print(train_df.shape)
    print(train_df.columns.tolist())
    print(train_df.head())

if __name__ == "__main__":
    args = parse_args()
    lines = [
        f"Raw data path: {args.raw_data}",
        f"Train dataset output path: {args.train_data}",
        f"Test dataset path: {args.test_data}",
        f"Test-train ratio: {args.test_train_ratio}",
    ]
    for line in lines:
        print(line)
    main(args)
     
