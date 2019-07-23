import cudf
import csv
import argparse
import logging
import torch.utils.dlpack as dlpack
from datetime import datetime
from rapidscyber.ml.provider.dga_detector import DGADetector
from rapidscyber.ml.dataset_api.dga_dataset import DGADataset
from rapidscyber.io.reader.fs_reader import FileSystemReader

log = logging.getLogger("train_dga_detector")


def train(epoch, dd, gdf):
    for iter in range(1, epoch + 1):
        print("*********")
        print("Epoch: %s" % (iter))
        print("*********")
        dd.train_model(gdf)
        now = datetime.now()
        output_filepath = "./rapidscyber/trained_models/rnn_classifier_{}.pth".format(
            now.strftime("%Y-%m-%d_%H_%M_%S")
        )
        print("saving model to filepath: %s" % (output_filepath))
        dd.save_model(output_filepath)


def main():

    input_path = args["input_path"]
    b_size = int(args["batch_size"])
    epoch = int(args["epoch"])
    log.info("input_path : %s" % (input_path))

    dd = DGADetector()
    dd.init_model()
    gdf = load_input(input_path)
    train(epoch, dd, gdf)


def load_input(input_path):
    config = {
        "input_path": input_path,
        "schema": ["domain", "types"],
        "delimiter": ",",
        "required_cols": ["domains", "types"],
        "dtype": ["str", "int"],
        "header": 0,
        "input_format": "text",
    }
    reader = FileSystemReader(config)
    gdf = reader.fetch_data()
    return gdf


def parse_cmd_args():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(description="Model training runner")
    ap.add_argument("input_path", help="input dataset for model training")
    ap.add_argument(
        "batch_size", help="dividing dataset into number of batches or sets or parts"
    )
    ap.add_argument(
        "epoch",
        help="One epoch is when an entire dataset is passed forward and backward through the neural network only once",
    )
    args = vars(ap.parse_args())
    return args


# execution starts here
if __name__ == "__main__":
    args = parse_cmd_args()
    main()
