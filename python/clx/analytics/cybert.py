import cudf
import cupy
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
import logging
from collections import defaultdict
from torch.utils.dlpack import from_dlpack
from transformers import BertForTokenClassification

log = logging.getLogger(__name__)

class Cybert:
    """
    Cyber log parsing using BERT. This class provides methods for loading models, prediction, and postprocessing.
    """
    def __init__(self):
        self._model = None
        self._label_map = {}
        self._vocab_lookup = {}
        self._vocabpath = self.__get_vocab_file_path()
        self._hashpath = self.__get_hash_table_path()

    def load_model(self, model_filepath, label_map_filepath, pretrained_model='bert-base-cased'):
        """
        Load cybert model.

        :param model_filepath: Filepath of the model (.pth or .bin) to be loaded
        :type model_filepath: str
        :param label_map_filepath: Filepath of the labels (.txt) to be used
        :type label_map_filepath: str
        :param pretrained_model: Name of pretrained model to be loaded from transformers repo, default is bert-base-cased
        :type pretrained_model: str

        Examples
        --------
        >>> from clx.analytics.cybert import Cybert
        >>> cyparse = Cybert()
        >>> cyparse.load_model('/path/to/model.pth', '/path/to/labels.txt')
        """
        with open(self._vocabpath) as f:
            for index, line in enumerate(f):
                self._vocab_lookup[index] = line.split()[0]
        
        with open(label_map_filepath) as f:
            for index, line in enumerate(f):
                self._label_map[index] = line.split()[0]  
                
        model_state_dict = torch.load(model_filepath)
        self._model = BertForTokenClassification.from_pretrained(pretrained_model, state_dict=model_state_dict, num_labels=len(self._label_map))
        self._model.cuda()
        self._model.eval()

    def preprocess(self, raw_data_df, stride_len=116, max_seq_len=128):
        """
        Preprocess and tokenize data for cybert model inference.
        
        :param raw_data_df: logs to be processed 
        :type model_filepath: cudf.Series
        :param stride_len: Max stride length for processing, default is 116
        :type stride_len: int
        :param max_seq_len: Max sequence length for processing, default is 128
        :type max_seq_len: int
        
        Examples
        --------
        >>> import cudf
        >>> from clx.analytics.cybert import Cybert
        >>> cyparse = Cybert()
        >>> cyparse.load_model('/path/to/model.pth', '/path/to/labels.txt') 
        >>> raw_data_df = cudf.Series(['Log event 1', 'Log event 2'])
        >>> input_ids, attention_masks, meta_data = cyparse.preprocess(raw_data_df)
        """
        raw_data_df = raw_data_df.str.replace('"','')
        raw_data_df = raw_data_df.str.replace('\\r', ' ')
        raw_data_df = raw_data_df.str.replace('\\t', ' ')
        raw_data_df = raw_data_df.str.replace('=','= ')
        raw_data_df = raw_data_df.str.replace('\\n', ' ')
        
        self._byte_count = raw_data_df.str.byte_count()
        self._max_num_chars = self._byte_count.sum()
        self._max_rows_tensor = int((self._byte_count/120).ceil().sum())
        
        input_ids, attention_mask, meta_data = raw_data_df.str.subword_tokenize(self._hashpath, 128, 116,
                                                                                max_num_strings=len(raw_data_df),\
                                                                                max_num_chars=self._max_num_chars,\
                                                                                max_rows_tensor=self._max_rows_tensor,\
                                                                                do_lower=False, do_truncate=False)
        num_rows = int(len(input_ids)/128)
        input_ids = from_dlpack((input_ids.reshape(num_rows,128).astype(cupy.float)).toDlpack())
        attention_mask = from_dlpack((attention_mask.reshape(num_rows,128).astype(cupy.float)).toDlpack())
        meta_data = meta_data.reshape(num_rows, 3)
            
        return input_ids.type(torch.long), attention_mask.type(torch.long), meta_data

    def inference(self, raw_data_df):
        """
        Cybert inference and postprocessing on dataset
        
        :param raw_data_df: logs to be processed 
        :type raw_data_df: cudf.Series
        :return: parsed_df
        :rtype: pandas.DataFrame
        :return: confidence_df
        :rtype: pandas.DataFrame

        Examples
        --------
        >>> import cudf
        >>> from clx.analytics.cybert import Cybert
        >>> cyparse = Cybert()
        >>> cyparse.load_model('/path/to/model.pth', '/path/to/labels.txt') 
        >>> raw_data_df = cudf.Series(['Log event 1', 'Log event 2'])
        >>> processed_df, confidence_df = cy.inference(raw_data_df)
        """
        input_ids, attention_masks, meta_data = self.preprocess(raw_data_df)
        with torch.no_grad():
            logits = self._model(input_ids, attention_masks)[0]
        logits = F.softmax(logits, dim=2)
        confidences, labels = torch.max(logits,2)
        infer_pdf = pd.DataFrame(meta_data).astype(int)
        infer_pdf.columns = ['doc','start','stop']
        infer_pdf['confidences'] = confidences.detach().cpu().numpy().tolist()
        infer_pdf['labels'] = labels.detach().cpu().numpy().tolist()
        infer_pdf['token_ids'] = input_ids.detach().cpu().numpy().tolist()  
        
        parsed_df, confidence_df = self.__postprocess(infer_pdf)
        return parsed_df, confidence_df
    
    def __get_hash_table_path(self):
        hash_table_path = "%s/resources/bert-base-cased-hash.txt" % os.path.dirname(
            os.path.realpath(__file__)
        )
        return hash_table_path
    
    def __get_vocab_file_path(self):
        vocab_file_path = "%s/resources/bert-base-cased-vocab.txt" % os.path.dirname(
            os.path.realpath(__file__)
        )
        return vocab_file_path
    
    def __postprocess(self, infer_pdf):
        #cut overlapping edges
        infer_pdf['confidences'] = infer_pdf.apply(lambda row: row['confidences'][row['start']:row['stop']], axis=1)
        infer_pdf['labels'] = infer_pdf.apply(lambda row: row['labels'][row['start']:row['stop']], axis=1)
        infer_pdf['token_ids'] = infer_pdf.apply(lambda row: row['token_ids'][row['start']:row['stop']], axis=1)
        
        #aggregated logs
        infer_pdf = infer_pdf.groupby('doc').agg({'token_ids': 'sum', 'confidences': 'sum', 'labels': 'sum'})
        
        #parse_by_label
        parsed_dfs = infer_pdf.apply(lambda row: self.__get_label_dicts(row), axis=1, result_type='expand')
        parsed_df = pd.DataFrame(parsed_dfs[0].tolist())
        confidence_df = pd.DataFrame(parsed_dfs[1].tolist())
        confidence_df = confidence_df.drop(['X'], axis=1).applymap(np.mean)
        
        #decode cleanup
        parsed_df = self.__decode_cleanup(parsed_df)
        return parsed_df, confidence_df

    def __get_label_dicts(self, row):
        token_dict = defaultdict(str)
        confidence_dict = defaultdict(list) 
        for label, confidence, token_id in zip(row['labels'], row['confidences'], row['token_ids']):
            text_token = self._vocab_lookup[token_id]
            if text_token[:2] != '##':  
             ## if not a subword use the current label, else use previous
                new_label = label
                new_confidence = confidence 
            token_dict[self._label_map[new_label]] = token_dict[self._label_map[new_label]] + ' ' + text_token
            confidence_dict[self._label_map[label]].append(new_confidence)
        return token_dict, confidence_dict

    def __decode_cleanup(self,df):
        return df.replace(' ##', '', regex=True) \
                 .replace(' : ', ':', regex=True) \
                 .replace('\[ ', '[', regex=True) \
                 .replace(' ]', ']', regex=True) \
                 .replace(' /', '/', regex=True) \
                 .replace('/ ', '/', regex=True) \
                 .replace(' - ', '-', regex=True) \
                 .replace(' \( ', ' (', regex=True)\
                 .replace(' \) ', ') ', regex=True)\
                 .replace('\+ ', '+', regex=True)\
                 .replace(' . ', '.', regex=True)