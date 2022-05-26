import logging

import cudf
from cudf.core.subword_tokenizer import SubwordTokenizer
import cupy
import torch
import torch.nn as nn
from torch.utils.dlpack import to_dlpack
from clx.analytics.sequence_classifier import SequenceClassifier
from clx.utils.data.dataloader import DataLoader
from clx.utils.data.dataset import Dataset
from transformers import AutoModelForSequenceClassification

log = logging.getLogger(__name__)


class MulticlassSequenceClassifier(SequenceClassifier):
    """
    Sequence Classifier using BERT. This class provides methods for training/loading BERT models, evaluation and prediction.
    """

    def init_model(self, model_or_path, num_labels):
        """
        Load model from huggingface or locally saved model.

        :param model_or_path: huggingface pretrained model name or directory path to model
        :type model_or_path: str
        :param num_labels: number of labels used only for multiclass classification
        :type num_labels: int

        Examples
        --------
        >>> from clx.analytics.multiclass_sequence_classifier import MulticlassSequenceClassifier
        >>> sc = MulticlassSequenceClassifier()

        >>> sc.init_model("bert-base-uncased", num_labels=4)  # huggingface pre-trained model

        >>> sc.init_model(model_path, num_labels=4) # locally saved model
        """
        self._model = AutoModelForSequenceClassification.from_pretrained(model_or_path, num_labels=num_labels)

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            self._model.cuda()
            self._model = nn.DataParallel(self._model)
        else:
            self._device = torch.device("cpu")

        self._tokenizer = SubwordTokenizer(self._hashpath, do_lower_case=True)

    def predict(self, input_data, max_seq_len=128, batch_size=32):
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
        >>> predictions = sc.predict(emails_test)
        """

        predict_gdf = cudf.DataFrame()
        predict_gdf["text"] = input_data

        predict_dataset = Dataset(predict_gdf)
        predict_dataloader = DataLoader(predict_dataset, batchsize=batch_size)

        preds = cudf.Series()

        self._model.eval()
        for df in predict_dataloader.get_chunks():
            b_input_ids, b_input_mask = self._bert_uncased_tokenize(df["text"], max_seq_len)
            with torch.no_grad():
                logits = self._model(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask
                )[0]

            logits = logits.type(torch.DoubleTensor).to(self._device)
            logits = cupy.fromDlpack(to_dlpack(logits))
            b_preds = cupy.argmax(logits, axis=1).flatten()
            b_preds = cudf.Series(b_preds)
            preds = preds.append(b_preds)

        return preds
