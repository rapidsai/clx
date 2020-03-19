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
    def __init__(self, max_num_logs=100, max_num_chars=100000, max_rows_tensor=1000, stride_len=48, max_seq_len=64):
        self.max_num_logs = max_num_logs
        self.max_num_chars = max_num_chars
        self.max_rows_tensor = max_rows_tensor
        self.stride_len = stride_len
        self.max_seq_len = max_seq_len
        currdir = os.path.dirname(os.path.abspath(__file__))
        self.hash_file = currdir + "/resources/hash_table.txt"
        
    def preprocess(self, raw_data_df):
        input_ids, attention_masks, meta_data = tokenizer.tokenize_df(raw_data_df, self.hash_file, max_sequence_length = self.max_seq_len,
                                                       stride=self.stride_len, do_lower=False, do_truncate=False, max_num_sentences=self.max_num_logs,
                                                       max_num_chars = self.max_num_chars, max_rows_tensor = self.max_rows_tensor)        
        return input_ids, attention_masks

    def load_model(self, model_filepath, label_map, num_labels):
        model_state_dict = torch.load(model_file_path)
        self.model = BertForTokenClassification.from_pretrained('bert-base-cased', state_dict=model_state_dict, num_labels=num_labels)
        self.max_seq_len = max_seq_len

    def inference(self):
        with torch.no_grad():
            logits = self.model(input_ids, attention_masks)[0]
        logits = F.softmax(logits, dim=2)
        confidences, labels = torch.max(logits,2)

        pdf = pd.DataFrame(meta_data.detach().cpu().numpy())
        pdf.columns = ['doc','start','stop']
        pdf['confidences'] = confidences.detach().cpu().numpy().tolist()
        pdf['labels'] = labels.detach().cpu().numpy().tolist()
        pdf['token_ids'] = input_ids.detach().cpu().numpy().tolist()

    def postprocessing(self):
        pdf['confidences'] = pdf.apply(lambda row: row['confidences'][row['start']:row['stop']], axis=1)
        pdf['labels'] = pdf.apply(lambda row: row['labels'][row['start']:row['stop']], axis=1)
        pdf['token_ids'] = pdf.apply(lambda row: row['token_ids'][row['start']:row['stop']], axis=1)
        parsed_df = pdf.apply(lambda row: tokens_by_label(row), axis=1)
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