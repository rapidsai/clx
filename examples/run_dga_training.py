# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example Usage: python train_dga_model.py \
                    --training-data benign_and_dga_domains.csv \
                    --output-dir trained_models \
                    --batch-size 10000 \
                    --epoch 2
"""
import os
import time
import cudf
import torch
import argparse
import numpy as np
from datetime import datetime
from clx.analytics.dga_detector import DGADetector
from clx.analytics.detector_dataset import DetectorDataset
from cuml.preprocessing.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score

LR = 0.001
N_LAYERS = 4
CHAR_VOCAB = 128
HIDDEN_SIZE = 100
N_DOMAIN_TYPE = 2


def train_and_eval(dd, train_dataset, test_dataset, epoch, output_dir):
    print("Initiating model training")
    create_dir(output_dir)
    max_accuracy = 0
    prev_model_file_path = ""
    for i in range(1, epoch + 1):
        print("---------")
        print("Epoch: %s" % (i))
        print("---------")
        dd.train_model(train_dataset)
        accuracy = dd.evaluate_model(test_dataset)
        now = datetime.now()
        output_filepath = (
            output_dir
            + "/"
            + "rnn_classifier_{}.bin".format(now.strftime("%Y-%m-%d_%H_%M_%S"))
        )
        if accuracy > max_accuracy:
            dd.save_model(output_filepath)
            max_accuracy = accuracy
            if prev_model_file_path:
                os.remove(prev_model_file_path)
            prev_model_file_path = output_filepath
    print(
        "Model with highest accuracy (%s) is stored to location %s"
        % (max_accuracy, prev_model_file_path)
    )
    return prev_model_file_path


def create_df(domain_df, type_series):
    df = cudf.DataFrame()
    df["domain"] = domain_df["domain"].reset_index(drop=True)
    df["type"] = type_series.reset_index(drop=True)
    return df


def create_dir(dir_path):
    print("Verify if directory `%s` is already exists." % (dir_path))
    if not os.path.exists(dir_path):
        print("Directory `%s` does not exists." % (dir_path))
        print("Creating directory `%s` to store trained models." % (dir_path))
        os.makedirs(dir_path)

def main():
    epoch = int(args["epoch"])
    input_filepath = args["training_data"]
    batch_size = int(args["batch_size"])
    output_dir = args["output_dir"]

    col_names = ["domain", "type"]
    dtypes = ["str", "int32"]
    input_df = cudf.read_csv(input_filepath, names=col_names, dtype=dtypes)
    domain_train, domain_test, type_train, type_test = train_test_split(
        input_df, "type", train_size=0.7
    )

    test_df = create_df(domain_test, type_test)
    train_df = create_df(domain_train, type_train)

    train_dataset = DetectorDataset(train_df, batch_size)
    test_dataset = DetectorDataset(test_df, batch_size)

    dd = DGADetector(lr=LR)
    dd.init_model(
        n_layers=N_LAYERS,
        char_vocab=CHAR_VOCAB,
        hidden_size=HIDDEN_SIZE,
        n_domain_type=N_DOMAIN_TYPE,
    )
    model_filepath = train_and_eval(dd, train_dataset, test_dataset, epoch, output_dir)


def parse_cmd_args():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(description="DGA model trainer")
    ap.add_argument(
        "--training-data", required=True, help="CSV with domain and type fields"
    )
    ap.add_argument(
        "--output-dir", required=True, help="output directory to save new model files"
    )
    ap.add_argument(
        "--batch-size",
        required=True,
        help="Dividing dataset into number of batches or sets or parts",
    )
    ap.add_argument(
        "--epoch",
        required=True,
        help="One epoch is when an entire dataset is passed forward and backward through the neural network only once",
    )
    args = vars(ap.parse_args())
    return args


# execution starts here
if __name__ == "__main__":
    args = parse_cmd_args()
    main()
    