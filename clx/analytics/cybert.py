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
    def __init__(self, max_num_logs=100, max_num_chars=100000, max_rows_tensor=1000, stride_len=48, max_seq_len=64):
        """Initialize the cybert log parser

        :param max_num_logs: Max number of logs for processing
        :type max_num_logs: int
        :param max_num_chars: Max number of chars for processing
        :type max_num_chars: int
        :param max_rows_tensor: Max row tensor for processing
        :type max_rows_tensor: int
        :param stride_len: Max stride length for processing
        :type stride_len: int
        :param max_seq_len: Max sequence length for processing
        :type max_seq_len: int
        """
        self._max_num_logs = max_num_logs
        self._max_num_chars = max_num_chars
        self._max_rows_tensor = max_rows_tensor
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
    def max_num_logs(self):
        """Max number of logs.

        :return: Max number of logs.
        :rtype: int
        """
        return self._max_num_logs

    @property
    def max_num_chars(self):
        """Max number of characters.

        :return: Max number of characters.
        :rtype: int
        """
        return self._max_num_chars

    @property
    def max_rows_tensor(self):
        """Max number of rows in tensor.

        :return: Max number of rows in tensor.
        :rtype: int
        """
        return self._max_rows_tensor

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

    def preprocess(self, raw_data_df):
        input_ids, attention_masks, meta_data = tokenizer.tokenize_df(raw_data_df, hash_file=self._hash_file, max_sequence_length = self._max_seq_len,
                                                                      stride=self._stride_len, do_lower=False, do_truncate=False, max_num_sentences=self._max_num_logs,
                                                                      max_num_chars = self._max_num_chars, max_rows_tensor = self._max_rows_tensor)
        return input_ids, attention_masks, meta_data

    def load_model(self, model_filepath, label_map_filepath):
        """
        Load cybert model.

        :param model_filepath: Filepath of the model to be loaded
        :type model_filepath: str
        :param label_map_filepath: YAML Filepath of the labels to be used
        :type label_map_filepath: str
        :param num_labels: Number of labels
        :type num_labels: int

        Examples
        --------
        >>> from clx.analytics.cybert import Cybert
        >>> cy = Cybert()
        >>> cy.load_model('/path/to/model', '/path/to/labels.yaml')
        """
        with open(label_map_filepath) as label_file:
            self._label_map = yaml.load(label_file, Loader=yaml.FullLoader)
        print(self._label_map)
        model_state_dict = torch.load(model_filepath)
        self.model = BertForTokenClassification.from_pretrained('bert-base-cased', state_dict=model_state_dict, num_labels=num_labels)
        self.model.cuda()
        self.model.eval()
        self._num_labels = len(self._label_map) + 1

    def inference(self, raw_data_df):
        """
        Cybert inference on dataset

        :param raw_data_df: Dataframe containing one column of raw logs
        :type raw_data_df: cudf.DataFrame
        :return: Processed inference data
        :rtype: cudf.DataFrame

        Examples
        --------
        >>> import cudf
        >>> from clx.analytics.cybert import Cybert
        >>> cy = Cybert()
        >>> cy.load_model('/path/to/model', '/path/to/labels.yaml', 21)
        >>> raw_df = cudf.DataFrame()
        >>> raw_df['logs'] = ['Log event']
        >>> processed_df = cy.inference(raw_df)
        """
        input_ids, attention_masks, meta_data = self.preprocess(raw_data_df)
        with torch.no_grad():
            logits = self.model(input_ids, attention_masks)[0]
        logits = F.softmax(logits, dim=2)
        confidences, labels = torch.max(logits,2)
        infer_pdf = pd.DataFrame(meta_data.detach().cpu().numpy())
        infer_pdf.columns = ['doc','start','stop']
        infer_pdf['confidences'] = confidences.detach().cpu().numpy().tolist()
        infer_pdf['labels'] = labels.detach().cpu().numpy().tolist()
        infer_pdf['token_ids'] = input_ids.detach().cpu().numpy().tolist()
        processed_infer_pdf = self.__postprocessing(infer_pdf)
        return processed_infer_pdf

    def __postprocessing(self, infer_input_pdf):
        infer_input_pdf['confidences'] = infer_input_pdf.apply(lambda row: row['confidences'][row['start']:row['stop']], axis=1)
        infer_input_pdf['labels'] = infer_input_pdf.apply(lambda row: row['labels'][row['start']:row['stop']], axis=1)
        infer_input_pdf['token_ids'] = infer_input_pdf.apply(lambda row: row['token_ids'][row['start']:row['stop']], axis=1)
        parsed_df = infer_input_pdf.apply(lambda row: self.__tokens_by_label(row), axis=1)
        parsed_df = pd.DataFrame(parsed_df.tolist())
        parsed_df = parsed_df.applymap(self.__decode_cleanup)
        return parsed_df

    def __tokens_by_label(self,row):
        token_dict = defaultdict(str)
        for token, label in zip(row['token_ids'], row['labels']):
            token_dict[self._label_map[label]] = token_dict[self._label_map[label]] + ' ' + self._vocab_dict[token]
        return token_dict

    def __confidence_by_label(self,row):
        confidence_dict = defaultdict(list)
        for label, confidence in zip(row['labels'], row['confidences']):
            confidence_dict[LABEL_MAP[label]].append(confidence)
        return confidence_dict

    def __decode_cleanup(self,row):
        return row.replace(' ##', '').replace(' . ', '.').replace(' : ', ':').replace(' / ','/')