# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
import os
import json
from azureml.core import Run, Model

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory')
    parser.add_argument('--model_info_output_path', type=str, help='Path to write model info JSON')
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')
    return args

def main(args):
    """Registers the trained model in Azure ML"""
    print("Registering", args.model_name)

    # Get current run and workspace
    run = Run.get_context()
    ws = run.experiment.workspace

    # Register the model directly with Azure ML
    model = Model.register(
        workspace=ws,
        model_path=args.model_path,
        model_name=args.model_name
    )

    # Write model info JSON
    print("Writing JSON")
    model_info = {"id": f"{model.name}:{model.version}"}
    output_path = os.path.join(args.model_info_output_path, "model_info.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as of:
        json.dump(model_info, of)

if __name__ == "__main__":
    args = parse_args()

    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Model info output path: {args.model_info_output_path}"
    ]
    for line in lines:
        print(line)

    main(args)


    
