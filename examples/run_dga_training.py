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
Example Usage: python run_dga_training.py \
                    --training-data benign_and_dga_domains.csv \
                    --output-dir trained_models \
                    --batch-size 10000 \
                    --epochs 2
"""
import os
import cudf
import torch
import argparse
from datetime import datetime
from clx.analytics.dga_detector import DGADetector

LR = 0.001
N_LAYERS = 4
CHAR_VOCAB = 128
HIDDEN_SIZE = 100
N_DOMAIN_TYPE = 2


def create_dir(dir_path):
    print("Verify if directory `%s` is already exists." % (dir_path))
    if not os.path.exists(dir_path):
        print("Directory `%s` does not exists." % (dir_path))
        print("Creating directory `%s` to store trained models." % (dir_path))
        os.makedirs(dir_path)

def main():
    epoch = int(args["epochs"])
    input_filepath = args["training_data"]
    batch_size = int(args["batch_size"])
    output_dir = args["output_dir"]
    # load input data to gpu memory
    input_df = cudf.read_csv(input_filepath)
    training_data = input_df['domain']
    labels = input_df['type']
    del input_df
    dd = DGADetector(lr=LR)
    dd.init_model(
        n_layers=N_LAYERS,
        char_vocab=CHAR_VOCAB,
        hidden_size=HIDDEN_SIZE,
        n_domain_type=N_DOMAIN_TYPE,
    )
    dd.train_model(training_data, labels, batch_size=batch_size, epochs=epochs, train_size=0.7)
    now = datetime.now()
    output_filepath = (
            output_dir
            + "/"
            + "rnn_classifier_{}.bin".format(now.strftime("%Y-%m-%d_%H_%M_%S"))
        )
    dd.save_model(output_filepath)

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
        "--epochs",
        required=True,
        help="One epoch is when an entire dataset is passed forward and backward through the neural network only once",
    )
    args = vars(ap.parse_args())
    return args


# execution starts here
if __name__ == "__main__":
    args = parse_cmd_args()
    main()
