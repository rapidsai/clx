import cudf
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from collections import defaultdict
from transformers import BertForTokenClassification
from clx.analytics import tokenizer


class Cybert:
    """
    Cybert log parser
    """

    def __init__(self, stride_len=116, max_seq_len=128):
        """Initialize the cybert log parser

        :param stride_len: Max stride length for processing
        :type stride_len: int
        :param max_seq_len: Max sequence length for processing
        :type max_seq_len: int
        """
        self._stride_len = stride_len
        self._max_seq_len = max_seq_len
        currdir = os.path.dirname(os.path.abspath(__file__))
        self._hash_file = currdir + "/resources/bert_hash_table.txt"
        self._vocab_file = currdir + "/resources/bert_vocab.txt"
        self._vocab_dict = {}
        with open(self._vocab_file) as f:
            for index, line in enumerate(f):
                self._vocab_dict[index] = line.split()[0]

    @property
    def stride_len(self):
        """Max stride length.

        :return: Max stride length.
        :rtype: int
        """
        return self._stride_len

    @property
    def max_seq_len(self):
        """Max sequence length.

        :return: Max sequence length.
        :rtype: int
        """
        return self._max_seq_len

    @property
    def hash_file(self):
        """Hash table file path.

        :return: Hash table file path
        :rtype: str
        """
        return self._hash_file

    @property
    def vocab(self):
        """Vocab file path.

        :return: Vocab file path.
        :rtype: str
        """
        return self._vocab_file

    @property
    def set_vocab(self, vocab_file_path):
        """
        Set source.

        :param source: vocab file path
        """
        self._vocab_file = vocab_file_path
        self._vocab_dict = {}
        with open(self._vocab_file) as f:
            for index, line in enumerate(f):
                self._vocab_dict[index] = line.split()[0]

    @property
    def label_map(self):
        return self._label_map

    def __preprocess(self, raw_data_col):
        raw_data_col = raw_data_col.str.replace('"', "")
        raw_data_col = raw_data_col.str.replace("\\r", " ")
        raw_data_col = raw_data_col.str.replace("\\t", " ")
        raw_data_col = raw_data_col.str.replace("=", "= ")
        raw_data_col = raw_data_col.str.replace("\\n", " ")
        max_num_logs = len(raw_data_col)
        byte_count = raw_data_col.str.byte_count()
        max_num_chars = byte_count.sum()
        max_rows_tensor = int((byte_count / 348).ceil().sum())
        input_ids, attention_masks, meta_data = tokenizer.tokenize_df(
            raw_data_col,
            hash_file=self._hash_file,
            max_sequence_length=self._max_seq_len,
            stride=self._stride_len,
            do_lower=False,
            do_truncate=False,
            max_num_sentences=max_num_logs,
            max_num_chars=max_num_chars,
            max_rows_tensor=max_rows_tensor,
        )
        return input_ids, attention_masks, meta_data

    def load_model(self, model_filepath, label_map_filepath):
        """
        Load cybert model.

        :param model_filepath: Filepath of the model to be loaded
        :type model_filepath: str
        :param label_map_filepath: Filepath of the labels to be used
        :type label_map_filepath: str

        Examples
        --------
        >>> from clx.analytics.cybert import Cybert
        >>> cy = Cybert()
        >>> cy.load_model('/path/to/model', '/path/to/labels.txt')
        """
        self._label_map_file = label_map_filepath
        self._label_map = {}
        with open(self._label_map_file) as f:
            for index, line in enumerate(f):
                self._label_map[index] = line.split()[0]
        model_state_dict = torch.load(model_filepath)
        self._num_labels = len(self._label_map)
        self.model = BertForTokenClassification.from_pretrained(
            "bert-base-cased", state_dict=model_state_dict, num_labels=self._num_labels
        )
        self.model.cuda()
        self.model.eval()

    def inference(self, raw_data_col):
        """
        Cybert inference on dataset

        :param raw_data_col: Series containing raw logs
        :type raw_data_col: cudf.Series
        :return: Processed inference data
        :rtype: cudf.DataFrame

        Examples
        --------
        >>> import cudf
        >>> from clx.analytics.cybert import Cybert
        >>> cy = Cybert()
        >>> cy.load_model('/path/to/model', '/path/to/labels.txt')
        >>> raw_df = cudf.DataFrame()
        >>> raw_df['logs'] = ['Log event']
        >>> processed_df = cy.inference(raw_df['logs'])
        """
        input_ids, attention_masks, meta_data = self.__preprocess(raw_data_col)
        with torch.no_grad():
            logits = self.model(input_ids, attention_masks)[0]
        logits = F.softmax(logits, dim=2)
        confidences, labels = torch.max(logits, 2)
        infer_pdf = pd.DataFrame(meta_data.detach().cpu().numpy())
        infer_pdf.columns = ["doc", "start", "stop"]
        infer_pdf["confidences"] = confidences.detach().cpu().numpy().tolist()
        infer_pdf["labels"] = labels.detach().cpu().numpy().tolist()
        infer_pdf["token_ids"] = input_ids.detach().cpu().numpy().tolist()

        # cut overlapping edges
        infer_pdf["confidences"] = infer_pdf.apply(
            lambda row: row["confidences"][row["start"] : row["stop"]], axis=1
        )
        infer_pdf["labels"] = infer_pdf.apply(
            lambda row: row["labels"][row["start"] : row["stop"]], axis=1
        )
        infer_pdf["token_ids"] = infer_pdf.apply(
            lambda row: row["token_ids"][row["start"] : row["stop"]], axis=1
        )

        # aggregated logs
        infer_pdf = infer_pdf.groupby("doc").agg(
            {"token_ids": "sum", "confidences": "sum", "labels": "sum"}
        )

        # parse_by_label
        parsed_dfs = infer_pdf.apply(
            lambda row: self.__parsed_by_label(row), axis=1, result_type="expand"
        )
        parsed_df = pd.DataFrame(parsed_dfs[0].tolist())
        confidence_df = pd.DataFrame(parsed_dfs[1].tolist())
        confidence_df = (
            confidence_df.drop(["X"], axis=1).applymap(np.mean).applymap(np.mean)
        )

        # decode cleanup
        parsed_df = self.__decode_cleanup(parsed_df)
        return parsed_df, confidence_df

    def __parsed_by_label(self, row):
        token_dict = defaultdict(str)
        confidence_dict = defaultdict(list)
        for label, confidence, token_id in zip(
            row["labels"], row["confidences"], row["token_ids"]
        ):
            text_token = self._vocab_dict[token_id]
            if text_token[:2] != "##":
                ## if not a subword use the current label, else use previous
                new_label = label
                new_confidence = confidence
            token_dict[self._label_map[new_label]] = (
                token_dict[self._label_map[new_label]] + " " + text_token
            )
            confidence_dict[self._label_map[label]].append(new_confidence)
        return token_dict, confidence_dict

    def __decode_cleanup(self, df):
        return (
            df.replace(" ##", "", regex=True)
            .replace(" : ", ":", regex=True)
            .replace("\[ ", "[", regex=True)
            .replace(" ]", "]", regex=True)
            .replace(" /", "/", regex=True)
            .replace("/ ", "/", regex=True)
            .replace(" - ", "-", regex=True)
            .replace(" \( ", " (", regex=True)
            .replace(" \) ", ") ", regex=True)
            .replace("\+ ", "+", regex=True)
            .replace(" . ", ".", regex=True)
        )
