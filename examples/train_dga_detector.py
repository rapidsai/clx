import os
import cudf
import argparse
import logging
from datetime import datetime
from clx.ml.provider.dga_detector import DGADetector

log = logging.getLogger(__name__)


def train(partitioned_dfs, dataset_len, epoch):
    dd = DGADetector()
    dd.init_model()
    if not os.path.exists("./trained_models"):
        os.makedirs("./trained_models")
    for i in range(1, epoch + 1):
        print("*********")
        print("Epoch: %s" % (i))
        print("*********")
        dd.train_model(partitioned_dfs, dataset_len)
        now = datetime.now()
        output_filepath = "./trained_models/rnn_classifier_{}.pth".format(
            now.strftime("%Y-%m-%d_%H_%M_%S")
        )
        print("saving model to filepath: %s" % (output_filepath))
        dd.save_model(output_filepath)


def main():
    epoch = int(args["epoch"])
    file_path = args["file_path"]
    batch_size = int(args["batch_size"])
    log.info("file_path : %s" % (file_path))

    df = cudf.read_csv(file_path, names=["domain", "type"])
    dataset_len = df["domain"].count()
    partitioned_dfs = df.partition_by_hash(["domain"], int(dataset_len / batch_size))
    train(partitioned_dfs, dataset_len, epoch)


def parse_cmd_args():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(description="Model training runner")
    ap.add_argument("file_path", help="input dataset for model training")
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
