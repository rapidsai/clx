import os
import time
import cudf
import argparse
import logging
from datetime import datetime
from clx.ml.dga_detector import DGADetector

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def train(
    dd,
    partitioned_dfs,
    dataset_len,
    test_partitioned_dfs,
    test_dataset_len,
    epoch,
    output_dir,
):
    """Initiate model training"""
    log.info("Verify if output directory `%s` is already exists." % (output_dir))
    if not os.path.exists(output_dir):
        log.info("Output directory `%s` does not exists." % (output_dir))
        log.info("Creating directory `%s` to store trained models." % (output_dir))
        os.makedirs(output_dir)
    max_accuracy = 0
    prev_model_file_path = ""
    for i in range(1, epoch + 1):
        log.info("**********")
        log.info("Epoch: %s" % (i))
        log.info("**********")
        dd.train_model(partitioned_dfs, dataset_len)
        accuracy = dd.evaluate_model(test_partitioned_dfs, test_dataset_len)
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


def main():
    epoch = int(args["epoch"])
    train_file_path = args["train_file_path"]
    test_file_path = args["test_file_path"]
    batch_size = int(args["batch_size"])
    output_dir = args["output_dir"]

    log.info("train_file_path : %s" % (train_file_path))
    log.info("test_file_path : %s" % (test_file_path))

    df = cudf.read_csv(train_file_path, names=["domain", "type"])
    dataset_len = df["domain"].count()

    test_df = cudf.read_csv(test_file_path, names=["domain", "type"])
    test_dataset_len = test_df["domain"].count()

    dd = DGADetector()
    dd.init_model()

    # https://github.com/rapidsai/cudf/issues/2861
    # Workaround for partition dataframe to small batches
    partitioned_dfs = pre_process(dd, df, dataset_len, batch_size)
    test_partitioned_dfs = pre_process(dd, test_df, test_dataset_len, batch_size)

    train(
        dd,
        partitioned_dfs,
        dataset_len,
        test_partitioned_dfs,
        test_dataset_len,
        epoch,
        output_dir,
    )


def pre_process(dd, df, dataset_len, batch_size):
    """Partition one dataframe to multiple small dataframes based on a given batch size."""
    df = dd.str2ascii(df, dataset_len)
    prev_chunk_offset = 0
    partitioned_dfs = []
    while prev_chunk_offset < dataset_len:
        curr_chunk_offset = prev_chunk_offset + batch_size
        chunk = df.iloc[prev_chunk_offset:curr_chunk_offset:1]
        partitioned_dfs.append(chunk)
        prev_chunk_offset = curr_chunk_offset
    return partitioned_dfs


def parse_cmd_args():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(description="Model training runner")
    ap.add_argument("train_file_path", help="input dataset for model training")
    ap.add_argument("test_file_path", help="input dataset for model testing")
    ap.add_argument("output_dir", help="ouput directory to store trained model")
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
