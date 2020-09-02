import cudf
import cupy
from cuml.preprocessing.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.dlpack import from_dlpack
import logging
from transformers import AdamW, BertForSequenceClassification
from tqdm import trange
import numpy as np
import os

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
        if self._model is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = BertForSequenceClassification.from_pretrained(model_or_path, num_labels=2)
        self._model.cuda()

    def train_model(self, emails, labels, learning_rate=3e-5, max_seq_len=128, batch_size=32, epochs=5):
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
        train_emails, validation_emails, train_labels, validation_labels = train_test_split(emails, 'label', train_size=0.8, random_state=2)

        # Tokenize training and validation
        # train_inputs, train_masks, _ = tokenizer.tokenize_df(train_emails, self._hashpath, max_sequence_length=max_seq_len, max_num_sentences=max_num_sentences, max_num_chars=max_num_chars, max_rows_tensor=max_rows_tensor, do_truncate=True)
        # validation_inputs, validation_masks, _ = tokenizer.tokenize_df(validation_emails, self._hashpath, max_sequence_length=max_seq_len, max_num_sentences=max_num_sentences, max_num_chars=max_num_chars, max_rows_tensor=max_rows_tensor, do_truncate=True)

        train_inputs, train_masks = self._bert_uncased_tokenize(train_emails.email, max_seq_len)
        validation_inputs, validation_masks = self._bert_uncased_tokenize(validation_emails.copy().email, max_seq_len)

        # convert labels to tensors
        train_labels = torch.tensor(train_labels.to_array())
        validation_labels = torch.tensor(validation_labels.to_array())

        # create dataloaders
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

        self._config_optimizer(learning_rate)

        self._model = self._train(train_dataloader, validation_dataloader, self._model, epochs)

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
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

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
        self._model.save_pretrained(save_to_path)

    def predict(self, emails, max_seq_len=128, batch_size=32):
        """
        Predict the class with the trained model

        :param emails: dataframe where each row contains one column holding email text
        :type emails: cudf.DataFrame
        :param max_seq_len: Limits the length of the sequence returned by tokenizer. If tokenized sentence is shorter than max_seq_len, output will be padded with 0s. If the tokenized sentence is longer than max_seq_len it will be truncated to max_seq_len.
        :type max_seq_len: int
        :param batch_size: batch size
        :type batch_size: int
        :return: predictions: predicted labels (0 or 1) for each email
        :rtype: cudf.Series

        Examples
        --------
        >>> from cuml.preprocessing.model_selection import train_test_split
        >>> emails_train, emails_test, labels_train, labels_test = train_test_split(train_emails_df, 'label', train_size=0.8)
        >>> phish_detect.train_model(emails_train, labels_train)
        >>> predictions = phish_detect(new_emails_df)
        """
        predict_inputs, predict_masks = self._bert_uncased_tokenize(emails.email, max_seq_len)

        predict_inputs = predict_inputs.type(torch.LongTensor)
        predict_data = TensorDataset(predict_inputs, predict_masks)
        predict_sampler = SequentialSampler(predict_data)
        predict_dataloader = DataLoader(predict_data, sampler=predict_sampler, batch_size=batch_size)

        self._model.eval()

        results = []
        for batch in predict_dataloader:

            batch = tuple(t.to(self._device) for t in batch)

            b_input_ids, b_input_mask = batch

            with torch.no_grad():

                logits = self._model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]

            logits = logits.detach().cpu().numpy()

            results.append(logits)

        preds = [item for sublist in results for item in sublist]
        preds = np.argmax(preds, axis=1).flatten()
        preds = cudf.Series(preds.tolist())

        return preds

    def _get_hash_table_path(self):
        hash_table_path = "%s/resources/bert-base-uncased-hash.txt" % os.path.dirname(
            os.path.realpath(__file__)
        )
        return hash_table_path

    def _config_optimizer(self, learning_rate):
        param_optimizer = list(self._model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        self._optimizer = AdamW(optimizer_grouped_parameters, learning_rate)

    def _flatten_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def _train(self, train_dataloader, validation_dataloader, model, epochs):
        train_loss_set = []  # Store loss and accuracy

        for _ in trange(epochs, desc="Epoch"):
            model.train()  # Enable training mode
            tr_loss = 0   # Tracking variables
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self._device) for t in batch)  # Add batch to GPU
                b_input_ids, b_input_mask, b_labels = batch  # Unpack the inputs from dataloader
                self._optimizer.zero_grad()  # Clear out the gradients
                loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)[0]  # forwardpass

                train_loss_set.append(loss.item())

                loss.backward()
                self._optimizer.step()  # update parameters
                tr_loss += loss.item()  # get a numeric value
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            print("Train loss: {}".format(tr_loss / nb_tr_steps))

            model.eval()  # Put model in evaluation mode to evaluate loss on the validation set

            eval_accuracy = 0
            nb_eval_steps = 0

            for batch in validation_dataloader:
                batch = tuple(t.to(self._device) for t in batch)

                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():  # Telling the model not to compute or store gradients, saving memory and speeding up validation
                    logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]  # Forward pass, calculate logit predictions
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                temp_eval_accuracy = self._flatten_accuracy(logits, label_ids)

                eval_accuracy += temp_eval_accuracy
                nb_eval_steps += 1

            print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

        return model

    def _evaluate_testset(self, test_dataloader):

        self._model.eval()

        tests, true_labels = [], []

        for batch in test_dataloader:

            batch = tuple(t.to(self._device) for t in batch)

            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():

                logits = self._model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tests.append(logits)
            true_labels.append(label_ids)

        return tests, true_labels

    def _bert_uncased_tokenize(self, strings, max_seq_len):
        """
        converts cudf.Series of strings to two torch tensors- token ids and attention mask with padding
        """
        num_strings = len(strings)
        token_ids, mask = strings.str.subword_tokenize(self._hashpath, max_length=max_seq_len, stride=max_seq_len, do_lower=False, do_truncate=True)[:2]

        # convert from cupy to torch tensor using dlpack
        input_ids = from_dlpack(token_ids.reshape(num_strings, max_seq_len).astype(cupy.float).toDlpack())
        attention_mask = from_dlpack(mask.reshape(num_strings, max_seq_len).astype(cupy.float).toDlpack())
        return input_ids.type(torch.long), attention_mask.type(torch.long)
