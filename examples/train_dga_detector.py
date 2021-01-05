import os
import time
import cudf
import argparse
import logging
from datetime import datetime
from clx.analytics.dga_detector import DGADetector
from clx.analytics.detector_dataset import DetectorDataset

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def train(dd, dataset, test_dataset, epoch, output_dir):
    """Initiate model training"""
    log.info("Verify if output directory `%s` is already exists." % (output_dir))
    if not os.path.exists(output_dir):
        log.info("Output directory `%s` does not exists." % (output_dir))
        log.info("Creating directory `%s` to store trained models." % (output_dir))
        os.makedirs(output_dir)
    start_time = time.time()
    max_accuracy = 0
    prev_model_file_path = ""
    for i in range(1, epoch + 1):
        log.info("**********")
        log.info("Epoch: %s" % (i))
        log.info("**********")

        dd.train_model(dataset)
        accuracy = dd.evaluate_model(test_dataset)
        now = datetime.now()
        output_filepath = (
            output_dir
            + "/"
            + "rnn_classifier_{}.pth".format(now.strftime("%Y-%m-%d_%H_%M_%S"))
        )

        if accuracy > max_accuracy:
            dd.save_model(output_filepath)
            max_accuracy = accuracy
            if prev_model_file_path:
                os.remove(prev_model_file_path)
            prev_model_file_path = output_filepath
    log.info(
        "Model with highest accuracy (%s) is stored to location %s"
        % (max_accuracy, prev_model_file_path)
    )
    end_time = time.time()
    log.info("Time taken for training a model : %s" % (end_time - start_time))


def main():
    epoch = int(args["epoch"])
    train_file_path = args["train_file_path"]
    test_file_path = args["test_file_path"]
    batch_size = int(args["batch_size"])
    output_dir = args["output_dir"]

    log.info("train_file_path : %s" % (train_file_path))
    log.info("test_file_path : %s" % (test_file_path))

    col_names = ["domain", "type"]
    dtypes = ["str", "int32"]
    df = cudf.read_csv(train_file_path, names=col_names, dtype=dtypes)
    test_df = cudf.read_csv(test_file_path, names=col_names, dtype=dtypes)

    dd = DGADetector()
    dd.init_model()

    dataset = DetectorDataset(df, batch_size)
    test_dataset = DetectorDataset(test_df, batch_size)
    del df
    del test_df
    train(dd, dataset, test_dataset, epoch, output_dir)


def parse_cmd_args():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(description="Model training runner")
    ap.add_argument("train_file_path", help="Input dataset for model training")
    ap.add_argument("test_file_path", help="Input dataset for model testing")
    ap.add_argument("output_dir", help="Output directory to store trained model")
    ap.add_argument(
        "batch_size", help="Dividing dataset into number of batches or sets or parts"
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
