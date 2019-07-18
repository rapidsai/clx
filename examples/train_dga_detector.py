import csv
import argparse
import logging
import torch
import torch.nn as nn
from datetime import datetime
from rapidscyber.ml.provider.dga_detector import DGADetector
from rapidscyber.ml.dataset_api.dga_dataset import DGADataset
from torch.utils.data import DataLoader

log = logging.getLogger("train_dga_detector")


def train(epoch, dd, data_loader):
    for iter in range(1, epoch + 1):
        dd.train_model(iter, data_loader)
        now = datetime.now()
        output_filepath = "./rapidscyber/trained_models/rnn_classifier_{}.pth".format(
            now.strftime("%Y-%m-%d_%H_%M_%S")
        )
        log.info("saving model to filepath: %s" % (output_filepath))
        dd.save_model(output_filepath)


def main():

    input_path = args["input_path"]
    b_size = int(args["batch_size"])
    epoch = int(args["epoch"])
    log.info("input_path : %s" % (input_path))

    rows = load_data(input_path)
    dataset = DGADataset(rows)
    data_loader = DataLoader(dataset=dataset, batch_size=b_size, shuffle=True)
    dd = DGADetector()
    dd.init_model()
    train(epoch, dd, data_loader)


def load_data(data_filepath):
    try:
        with open(data_filepath, "rt") as f:
            reader = csv.reader(f)
            rows = list(reader)
        return rows
    except:
        log.error("Error occured while reading a file.")
        raise


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
