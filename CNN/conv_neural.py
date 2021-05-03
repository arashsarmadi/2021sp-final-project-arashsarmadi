import argparse

import os

from .mod_train import train_model
from .mod_test import test_model

# Source code from https://github.com/BadreeshShetty/OCT-Retinal/blob/master/OCT-Retinal%205%20CNN.ipynb


parser = argparse.ArgumentParser("Insetr image path")
parser.add_argument("-d", "--data", default="data/OCTReduced")
parser.add_argument("-o", "--out")
parser.add_argument("-a", "--action", default="train")
parser.add_argument("-l", "--model", default="train")


def main(args=None):
    args = parser.parse_args(args=args)

    img_rows, img_cols = 224, 224

    train_data_dir = os.path.join(args.data, "train")
    validation_data_dir = os.path.join(args.data, "val")
    test_data_dir = os.path.join(args.data, "test")
    action = args.action

    output = args.out
    model_path = args.model
    output_path = output.rstrip(output.split("/")[-1])

    if action == "train":
        train_model(
            train_data_dir, validation_data_dir, img_rows, img_cols, output_path, output
        )
    elif action == "test":
        test_model(test_data_dir, output_path, model_path, output)
    else:
        raise ValueError("Invalid action")
