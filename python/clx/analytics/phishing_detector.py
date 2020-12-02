import logging
import os

import cudf
import cupy
import numpy as np
import torch
import torch.nn as nn
import gc
from cuml.preprocessing.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torch.utils.dlpack import from_dlpack, to_dlpack
from tqdm import trange
from transformers import AdamW, BertForSequenceClassification

log = logging.getLogger(__name__)


class PhishingDetector:
    """
    Phishing detection using BERT. This class provides methods for training/loading BERT models, evaluation and prediction.
    """

    def __init__(self):
        self._device = None
        self._model = None
        self._optimizer = None
        self._hashpath = self._get_hash_table_path()

    def init_model(self, model_or_path="bert-base-uncased"):
        """
        Load a pretrained BERT model. Default is bert-base-uncased.

        :param model_or_path: directory path to model, default is bert-base-uncased
        :type input_file: str

        Examples
        --------
        >>> from clx.analytics.phishing_detector import PhishingDetector
        >>> phish_detect = PhishingDetector()

        >>> phish_detect.init_model()  # bert-base-uncased

        >>> phish_detect.init_model(model_path)
        """
        self._model = BertForSequenceClassification.from_pretrained(
            model_or_path, num_labels=2
        )
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            self._model.cuda()
            self._model = nn.DataParallel(self._model)
        else:
            self._device = torch.device("cpu")

    def train_model(
        self,
        emails,
        labels,
        learning_rate=3e-5,
        max_seq_len=128,
        batch_size=32,
        epochs=5,
    ):
        """
        Train the classifier

        :param emails: dataframe where each row contains one column holding email text
        :type emails: cudf.DataFrame
        :param labels: series holding labels for each row in email dataframe
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
        >>> phish_detect.train_model(emails_train, labels_train)
        """
        emails["label"] = labels
        (
            train_emails,
            validation_emails,
            train_labels,
            validation_labels,
        ) = train_test_split(emails, "label", train_size=0.8, random_state=2)

        train_emails["label"] = train_labels
        validation_emails["label"] = validation_labels

        train_dataset = self._get_partitioned_dfs(train_emails, batch_size)
        validation_dataset = self._get_partitioned_dfs(validation_emails, batch_size)

        self._config_optimizer(learning_rate)

        for _ in trange(epochs, desc="Epoch"):
            self._model.train()  # Enable training mode
            tr_loss = 0  # Tracking variables
            nb_tr_examples, nb_tr_steps = 0, 0

            # iterate through partitioned training dataset
            for b_df in train_dataset:
                b_input_ids, b_input_mask = self._bert_uncased_tokenize(
                    b_df.email, max_seq_len
                )
                b_labels = torch.tensor(b_df["label"].to_array()).to(self._device)
                self._optimizer.zero_grad()  # Clear out the gradients
                loss = self._model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )[
                    0
                ]  # forwardpass

                loss.backward()
                self._optimizer.step()  # update parameters
                tr_loss += loss.item()  # get a numeric value
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            print("Train loss: {}".format(tr_loss / nb_tr_steps))

        self._model.eval()  # Put model in evaluation mode to evaluate loss on the validation set

        eval_accuracy = 0
        nb_eval_steps = 0

        # iterate through partitioned validation dataset
        for b_df in validation_dataset:
            b_input_ids, b_input_mask = self._bert_uncased_tokenize(
                b_df.email, max_seq_len
            )
            b_labels = torch.tensor(b_df["label"].to_array()).to(self._device)

            with torch.no_grad():  # Telling the model not to compute or store gradients, saving memory and speeding up validation
                logits = self._model(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask
                )[
                    0
                ]  # Forward pass, calculate logit predictions
            logits = cupy.fromDlpack(to_dlpack(logits))
            label_ids = cupy.fromDlpack(to_dlpack(b_labels))

            temp_eval_accuracy = self._flatten_accuracy(logits, label_ids)

            eval_accuracy += temp_eval_accuracy
            nb_eval_steps += 1

        print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

    def evaluate_model(self, emails, labels, max_seq_len=128, batch_size=32):
        """
        Evaluate trained BERT model

        :param emails: dataframe where each row contains one column holding email text
        :type emails: cudf.Dataframe
        :param labels: series holding labels for each row in email dataframe
        :type labels: cudf.Series
        :param max_seq_len: Limits the length of the sequence returned by tokenizer. If tokenized sentence is shorter than max_seq_len, output will be padded with 0s. If the tokenized sentence is longer than max_seq_len it will be truncated to max_seq_len.
        :type max_seq_len: int
        :param batch_size: batch size
        :type batch_size: int

        Examples
        --------
        >>> from cuml.preprocessing.model_selection import train_test_split
        >>> emails_train, emails_test, labels_train, labels_test = train_test_split(train_emails_df, 'label', train_size=0.8)
        >>> phish_detect.evaluate_model(emails_test, labels_test)
        """
        test_inputs, test_masks = self._bert_uncased_tokenize(emails.email, max_seq_len)

        test_labels = torch.tensor(labels.to_array())
        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(
            test_data, sampler=test_sampler, batch_size=batch_size
        )

        tests, true_labels = self._evaluate_testset(test_dataloader)

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
        >>> phish_detect.train_model(emails_train, labels_train)
        >>> phish_detect.save_model()
        """

        self._model.module.save_pretrained(save_to_path)

    def predict(self, emails, max_seq_len=128, threshold=0.5):
        """
        Predict the class with the trained model

        :param emails: series where each element is text from single email
        :type emails: cudf.Series
        :param max_seq_len: Limits the length of the sequence returned by tokenizer. If tokenized sentence is shorter than max_seq_len, output will be padded with 0s. If the tokenized sentence is longer than max_seq_len it will be truncated to max_seq_len.
        :type max_seq_len: int
        :param batch_size: batch size
        :type batch_size: int
        :param threshold: results with probabilities higher than this will be labeled as positive
        :type threshold: float
        :return: predictions: predicted labels (False or True) for each email
        :rtype: cudf.Series

        Examples
        --------
        >>> from cuml.preprocessing.model_selection import train_test_split
        >>> emails_train, emails_test, labels_train, labels_test = train_test_split(train_emails_df, 'label', train_size=0.8)
        >>> phish_detect.train_model(emails_train, labels_train)
        >>> predictions = phish_detect.predict(new_emails, threshold=0.8)
        """
        predict_inputs, predict_masks = self._bert_uncased_tokenize(emails, max_seq_len)
        predict_inputs = predict_inputs.type(torch.LongTensor).to(self._device)
        predict_masks = predict_masks.to(self._device)
        with torch.no_grad():
            logits = self._model(
                predict_inputs, token_type_ids=None, attention_mask=predict_masks
            )[0]
            probs = torch.sigmoid(logits[:, 1])
            preds = probs.ge(threshold)

        probs = cudf.io.from_dlpack(to_dlpack(probs))
        preds = cudf.io.from_dlpack(to_dlpack(preds))

        torch.cuda.empty_cache()
        gc.collect()

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

    def _evaluate_testset(self, test_dataloader):

        self._model.eval()

        tests, true_labels = [], []

        for batch in test_dataloader:

            batch = tuple(t.to(self._device) for t in batch)

            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():

                logits = self._model(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask
                )[0]

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            tests.append(logits)
            true_labels.append(label_ids)

        return tests, true_labels

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

    def _get_partitioned_dfs(self, df, batch_size):
        dataset_len = df.shape[0]
        prev_chunk_offset = 0
        partitioned_dfs = []
        while prev_chunk_offset < dataset_len:
            curr_chunk_offset = prev_chunk_offset + batch_size
            chunk = df.iloc[prev_chunk_offset:curr_chunk_offset:1]
            partitioned_dfs.append(chunk)
            prev_chunk_offset = curr_chunk_offset
        return partitioned_dfs
