from transformers import BertForTokenClassification
import cudf
import numpy as np
import torch
import torch.nn.functional as F
import tokenizer
import pandas as pd
from collections import defaultdict
import os

class Cybert:
    """
    Cybert log parser 
    """
    def __init__(self, max_num_logs=100, max_num_chars=100000, max_rows_tensor=1000, stride_len=48, max_seq_len=64):
        """
        
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
        self.hash_file = currdir + "/resources/hash_table.txt"

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
    def max_stride_len(self):
        """Max stride length.
        
        :return: Max stride length.
        :rtype: int
        """
        return self._max_stride_len

    @property
    def max_seq_len(self):
        """Max sequence length.
        
        :return: Max sequence length.
        :rtype: int
        """
        return self._max_seq_len

    def preprocess(self, raw_data_df):
        input_ids, attention_masks, meta_data = tokenizer.tokenize_df(raw_data_df, self.hash_file, max_sequence_length = self.max_seq_len,
                                                       stride=self.stride_len, do_lower=False, do_truncate=False, max_num_sentences=self.max_num_logs,
                                                       max_num_chars = self.max_num_chars, max_rows_tensor = self.max_rows_tensor)        
        return input_ids, attention_masks

    def load_model(self, model_filepath, label_map, num_labels):
        model_state_dict = torch.load(model_file_path)
        self.model = BertForTokenClassification.from_pretrained('bert-base-cased', state_dict=model_state_dict, num_labels=num_labels)
        self.max_seq_len = max_seq_len

    def inference(self, input_ids, attention_masks):
        with torch.no_grad():
            logits = self.model(input_ids, attention_masks)[0]
        logits = F.softmax(logits, dim=2)
        confidences, labels = torch.max(logits,2)

        infer_pdf = pd.DataFrame(meta_data.detach().cpu().numpy())
        infer_pdf.columns = ['doc','start','stop']
        infer_pdf['confidences'] = confidences.detach().cpu().numpy().tolist()
        infer_pdf['labels'] = labels.detach().cpu().numpy().tolist()
        infer_pdf['token_ids'] = input_ids.detach().cpu().numpy().tolist()
        processed_infer_pdf = self.postprocessing(infer_pdf)
        return infer_pdf

    def postprocessing(self, infer_pdf):
        infer_pdf['confidences'] = infer_pdf.apply(lambda row: row['confidences'][row['start']:row['stop']], axis=1)
        infer_pdf['labels'] = infer_pdf.apply(lambda row: row['labels'][row['start']:row['stop']], axis=1)
        infer_pdf['token_ids'] = infer_pdf.apply(lambda row: row['token_ids'][row['start']:row['stop']], axis=1)
        parsed_df = infer_pdf.apply(lambda row: self.__tokens_by_label(row), axis=1)
        parsed_df = pd.DataFrame(parsed_df.tolist())
        parsed_df = parsed_df.applymap(self.__decode_cleanup)
        conf_df = pdf.apply(lambda row: self.__confidence_by_label(row), axis=1)
        conf_df = pd.DataFrame(conf_df.tolist())

    def __tokens_by_label(self,row):
        token_dict = defaultdict(str)
        for token, label in zip(row['token_ids'], row['labels']):
            token_dict[LABEL_MAP[label]] = token_dict[LABEL_MAP[label]] + ' ' + vocab_dict[token]
        return token_dict

    def __confidence_by_label(self,row):
        confidence_dict = defaultdict(list)
        for label, confidence in zip(row['labels'], row['confidences']):
            confidence_dict[LABEL_MAP[label]].append(confidence)    
        return confidence_dict

    def __decode_cleanup(self,string):
        return string.replace(' ##', '').replace(' . ', '.').replace(' : ', ':').replace(' / ','/')