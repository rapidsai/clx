import csv
import torch
import argparse
import logging
import torch.nn as nn
from datetime import datetime
from rapidscyber.ml.manager.rnn_classifier_builder import RNNClassifierBuilder
from rapidscyber.ml.manager.rnn_classifier_service import RNNClassifierService
from rapidscyber.ml.dataset_api.dga_dataset import DGADataset
from torch.utils.data import DataLoader

log = logging.getLogger("train_rnn_classifier")


def train(epoch, rnn_classifier_service):
    for iter in range(1, epoch + 1):
        model, total_loss = rnn_classifier_service.train_model(iter)
        now = datetime.now()
        output_file = "./rapidscyber/trained_models/rnn_classifier_{}.pth".format(
            now.strftime("%Y-%m-%d_%H_%M_%S")
        )
        log.info("saving model to file: %s" % (output_file))
        torch.save(model, output_file)


def main():

    n_domain_type = 2
    hidden_size = 100
    n_layers = 3
    char_vocab = 128
    learning_rate = 0.001

    input_path = args["input_path"]
    b_size = int(args["batch_size"])
    epoch = int(args["epoch"])

    log.info("input_path : %s" % (input_path))
    rows = load_data(input_path)
    dataset = DGADataset()
    dataset.set_attributes(rows)
    train_loader = DataLoader(dataset=dataset, batch_size=b_size, shuffle=True)

    builder = RNNClassifierBuilder(char_vocab, hidden_size, n_domain_type, n_layers)
    model = builder.build_model()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    rnn_classifier_service = RNNClassifierService(
        model, optimizer, criterion, train_loader=train_loader, b_size=b_size
    )
    train(epoch, rnn_classifier_service)


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
