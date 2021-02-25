import logging
import os

import cudf
import cupy
import numpy as np
import torch
import torch.nn as nn
from clx.utils.data.dataloader import DataLoader
from clx.utils.data.dataset import Dataset
from sklearn.metrics import accuracy_score
from torch.utils.dlpack import from_dlpack, to_dlpack
from tqdm import trange
from transformers import AutoModelForSequenceClassification, AdamW

log = logging.getLogger(__name__)


class SequenceClassifier:
    """
    Sequence Classifier using BERT. This class provides methods for training/loading BERT models, evaluation and prediction.
    """

    def __init__(self):
        self._device = None
        self._model = None
        self._optimizer = None
        self._hashpath = self._get_hash_table_path()

    def init_model(self, model_or_path):
        """
        Load model from huggingface or locally saved model.

        :param model_or_path: huggingface pretrained model name or directory path to model
        :type model_or_path: str

        Examples
        --------
        >>> from clx.analytics.sequence_classifier import SequenceClassifier
        >>> sc = SequenceClassifier()

        >>> sc.init_model("bert-base-uncased")  # huggingface pre-trained model

        >>> sc.init_model(model_path) # locally saved model
        """
        self._model = AutoModelForSequenceClassification.from_pretrained(model_or_path)

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            self._model.cuda()
            self._model = nn.DataParallel(self._model)
        else:
            self._device = torch.device("cpu")

    def train_model(
        self,
        train_data,
        labels,
        learning_rate=3e-5,
        max_seq_len=128,
        batch_size=32,
        epochs=5,
    ):
        """
        Train the classifier

        :param train_data: text data for training
        :type train_data: cudf.Series
        :param labels: labels for each element in train_data
        :type labels: cudf.Series
        :param learning_rate: learning rate
        :type learning_rate: float
        :param max_seq_len: Limits the length of the sequence returned by tokenizer. If tokenized sentence is shorter than max_seq_len, output will be padded with 0s. If the tokenized sentence is longer than max_seq_len it will be truncated to max_seq_len.
        :type max_seq_len: int
        :param batch_size: batch size
        :type batch_size: int
        :param epoch: epoch, default is 5
        :type epoch: int

        Examples
        --------
        >>> from cuml.preprocessing.model_selection import train_test_split
        >>> emails_train, emails_test, labels_train, labels_test = train_test_split(train_emails_df, 'label', train_size=0.8)
        >>> sc.train_model(emails_train, labels_train)
        """
        train_gdf = cudf.DataFrame()
        train_gdf["text"] = train_data
        train_gdf["label"] = labels

        train_dataset = Dataset(train_gdf)
        train_dataloader = DataLoader(train_dataset, batchsize=batch_size)

        self._config_optimizer(learning_rate)
        self._model.train()  # Enable training mode

        for _ in trange(epochs, desc="Epoch"):
            tr_loss = 0   # Tracking variables
            nb_tr_examples, nb_tr_steps = 0, 0
            for df in train_dataloader.get_chunks():
                b_labels = df["label"]
                b_input_ids, b_input_mask = self._bert_uncased_tokenize(df["text"], max_seq_len)
                b_labels = torch.tensor(df["label"].to_array())
                self._optimizer.zero_grad()  # Clear out the gradients
                loss = self._model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)[0]  # forwardpass

                loss.sum().backward()
                self._optimizer.step()  # update parameters
                tr_loss += loss.sum().item()  # get a numeric value
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            print("Train loss: {}".format(tr_loss / nb_tr_steps))

    def evaluate_model(self, test_data, labels, max_seq_len=128, batch_size=32):
        """
        Evaluate trained model

        :param test_data: test data to evaluate model
        :type test_data: cudf.Series
        :param labels: labels for each element in test_data
        :type labels: cudf.Series
        :param max_seq_len: Limits the length of the sequence returned by tokenizer. If tokenized sentence is shorter than max_seq_len, output will be padded with 0s. If the tokenized sentence is longer than max_seq_len it will be truncated to max_seq_len.
        :type max_seq_len: int
        :param batch_size: batch size
        :type batch_size: int

        Examples
        --------
        >>> from cuml.preprocessing.model_selection import train_test_split
        >>> emails_train, emails_test, labels_train, labels_test = train_test_split(train_emails_df, 'label', train_size=0.8)
        >>> sc.evaluate_model(emails_test, labels_test)
        """
        test_gdf = cudf.DataFrame()
        test_gdf["text"] = test_data
        test_gdf["label"] = labels

        test_dataset = Dataset(test_gdf)
        test_dataloader = DataLoader(test_dataset, batchsize=batch_size)

        tests, true_labels = [], []
        for df in test_dataloader.get_chunks():
            b_labels = df["label"]
            b_input_ids, b_input_mask = self._bert_uncased_tokenize(df["text"], max_seq_len)
            b_labels = torch.tensor(df["label"].to_array())
            with torch.no_grad():

                logits = self._model(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask
                )[0]

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            tests.append(logits)
            true_labels.append(label_ids)

        flat_tests = [item for sublist in tests for item in sublist]
        flat_tests = np.argmax(flat_tests, axis=1).flatten()
        flat_true_labels = [item for sublist in true_labels for item in sublist]

        accuracy = accuracy_score(flat_true_labels, flat_tests)

        return accuracy

    def save_model(self, save_to_path="."):
        """
        Save trained model

        :param save_to_path: directory path to save model, default is current directory
        :type save_to_path: str

        Examples
        --------
        >>> from cuml.preprocessing.model_selection import train_test_split
        >>> emails_train, emails_test, labels_train, labels_test = train_test_split(train_emails_df, 'label', train_size=0.8)
        >>> sc.train_model(emails_train, labels_train)
        >>> sc.save_model()
        """

        self._model.module.save_pretrained(save_to_path)

    def predict(self, input_data, max_seq_len=128, batch_size=32, threshold=0.5):
        """
        Predict the class with the trained model

        :param input_data: input text data for prediction
        :type input_data: cudf.Series
        :param max_seq_len: Limits the length of the sequence returned by tokenizer. If tokenized sentence is shorter than max_seq_len, output will be padded with 0s. If the tokenized sentence is longer than max_seq_len it will be truncated to max_seq_len.
        :type max_seq_len: int
        :param batch_size: batch size
        :type batch_size: int
        :param threshold: results with probabilities higher than this will be labeled as positive
        :type threshold: float
        :return: predictions, probabilities: predictions are labels (0 or 1) based on minimum threshold
        :rtype: cudf.Series, cudf.Series

        Examples
        --------
        >>> from cuml.preprocessing.model_selection import train_test_split
        >>> emails_train, emails_test, labels_train, labels_test = train_test_split(train_emails_df, 'label', train_size=0.8)
        >>> sc.train_model(emails_train, labels_train)
        >>> predictions = sc.predict(emails_test, threshold=0.8)
        """

        predict_gdf = cudf.DataFrame()
        predict_gdf["text"] = input_data

        predict_dataset = Dataset(predict_gdf)
        predict_dataloader = DataLoader(predict_dataset, batchsize=batch_size)

        preds = cudf.Series()
        probs = cudf.Series()
        for df in predict_dataloader.get_chunks():
            b_input_ids, b_input_mask = self._bert_uncased_tokenize(df["text"], max_seq_len)
            with torch.no_grad():
                logits = self._model(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask
                )[0]
                b_probs = torch.sigmoid(logits[:, 1])
                b_preds = b_probs.ge(threshold)

            b_probs = cudf.io.from_dlpack(to_dlpack(b_probs))
            b_preds = cudf.io.from_dlpack(to_dlpack(b_preds))
            preds = preds.append(b_preds)
            probs = probs.append(b_probs)

        return preds, probs

    def _get_hash_table_path(self):
        hash_table_path = "%s/resources/bert-base-uncased-hash.txt" % os.path.dirname(
            os.path.realpath(__file__)
        )
        return hash_table_path

    def _config_optimizer(self, learning_rate):
        param_optimizer = list(self._model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]
        self._optimizer = AdamW(optimizer_grouped_parameters, learning_rate)

    def _flatten_accuracy(self, preds, labels):
        pred_flat = cupy.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return cupy.sum(pred_flat == labels_flat) / len(labels_flat)

    def _bert_uncased_tokenize(self, strings, max_seq_len):
        """
        converts cudf.Series of strings to two torch tensors- token ids and attention mask with padding
        """
        num_strings = len(strings)
        token_ids, mask = strings.str.subword_tokenize(
            self._hashpath,
            max_length=max_seq_len,
            stride=max_seq_len,
            do_lower=False,
            do_truncate=True,
        )[:2]

        # convert from cupy to torch tensor using dlpack
        input_ids = from_dlpack(
            token_ids.reshape(num_strings, max_seq_len).astype(cupy.float).toDlpack()
        )
        attention_mask = from_dlpack(
            mask.reshape(num_strings, max_seq_len).astype(cupy.float).toDlpack()
        )
        return input_ids.type(torch.long), attention_mask.type(torch.long)
