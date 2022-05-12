# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    BertForTokenClassification,
    DistilBertForTokenClassification,
    ElectraForTokenClassification,
)

from cudf.core.subword_tokenizer import SubwordTokenizer


log = logging.getLogger(__name__)

ARCH_MAPPING = {
    "BertForTokenClassification": BertForTokenClassification,
    "DistilBertForTokenClassification": DistilBertForTokenClassification,
    "ElectraForTokenClassification": ElectraForTokenClassification,
}

MODEL_MAPPING = {
    "BertForTokenClassification": "bert-base-cased",
    "DistilBertForTokenClassification": "distilbert-base-cased",
    "ElectraForTokenClassification": "rapids/electra-small-discriminator",
}


class Cybert:
    """
    Cyber log parsing using BERT, DistilBERT, or ELECTRA. This class provides methods
    for loading models, prediction, and postprocessing.
    """

    def __init__(self):
        self._model = None
        self._label_map = {}
        resources_dir = "%s/resources" % os.path.dirname(os.path.realpath(__file__))
        vocabpath = "%s/bert-base-cased-vocab.txt" % resources_dir
        self._vocab_lookup = {}
        with open(vocabpath) as f:
            for index, line in enumerate(f):
                self._vocab_lookup[index] = line.split()[0]
        self._hashpath = "%s/bert-base-cased-hash.txt" % resources_dir

        self.tokenizer = SubwordTokenizer(self._hashpath, do_lower_case=False)

    def load_model(self, model_filepath, config_filepath):
        """
        Load cybert model.

        :param model_filepath: Filepath of the model (.pth or .bin) to be loaded
        :type model_filepath: str
        :param config_filepath: Config file (.json) to be used
        :type config_filepath: str

        Examples
        --------
        >>> from clx.analytics.cybert import Cybert
        >>> cyparse = Cybert()
        >>> cyparse.load_model('/path/to/model.bin', '/path/to/config.json')
        """

        with open(config_filepath) as f:
            config = json.load(f)
        model_arch = config["architectures"][0]
        self._label_map = {int(k): v for k, v in config["id2label"].items()}
        self._model = ARCH_MAPPING[model_arch].from_pretrained(
            model_filepath, config=config_filepath,
        )
        self._model.cuda()
        self._model.eval()
        self._model = nn.DataParallel(self._model)

    def preprocess(self, raw_data_col, stride_len=116, max_seq_len=128):
        """
        Preprocess and tokenize data for cybert model inference.

        :param raw_data_col: logs to be processed
        :type raw_data_col: cudf.Series
        :param stride_len: Max stride length for processing, default is 116
        :type stride_len: int
        :param max_seq_len: Max sequence length for processing, default is 128
        :type max_seq_len: int

        Examples
        --------
        >>> import cudf
        >>> from clx.analytics.cybert import Cybert
        >>> cyparse = Cybert()
        >>> cyparse.load_model('/path/to/model.pth', '/path/to/config.json')
        >>> raw_df = cudf.Series(['Log event 1', 'Log event 2'])
        >>> input_ids, attention_masks, meta_data = cyparse.preprocess(raw_df)
        """
        raw_data_col = raw_data_col.str.replace('"', "")
        raw_data_col = raw_data_col.str.replace("\\r", " ")
        raw_data_col = raw_data_col.str.replace("\\t", " ")
        raw_data_col = raw_data_col.str.replace("=", "= ")
        raw_data_col = raw_data_col.str.replace("\\n", " ")

        output = self.tokenizer(
            raw_data_col,
            max_length=128,
            stride=12,
            max_num_rows=len(raw_data_col),
            truncation=False,
            add_special_tokens=False,
            return_tensors="pt",
        )

        input_ids = output["input_ids"].type(torch.long)
        attention_masks = output["attention_mask"].type(torch.long)
        meta_data = output["metadata"]

        return input_ids, attention_masks, meta_data

    def inference(self, raw_data_col, batch_size=160):
        """
        Cybert inference and postprocessing on dataset

        :param raw_data_col: logs to be processed
        :type raw_data_col: cudf.Series
        :param batch_size: Log data is processed in batches using a Pytorch dataloader.
        The batch size parameter refers to the batch size indicated in torch.utils.data.DataLoader.
        :type batch_size: int
        :return: parsed_df
        :rtype: pandas.DataFrame
        :return: confidence_df
        :rtype: pandas.DataFrame

        Examples
        --------
        >>> import cudf
        >>> from clx.analytics.cybert import Cybert
        >>> cyparse = Cybert()
        >>> cyparse.load_model('/path/to/model.pth', '/path/to/config.json')
        >>> raw_data_col = cudf.Series(['Log event 1', 'Log event 2'])
        >>> processed_df, confidence_df = cy.inference(raw_data_col)
        """
        input_ids, attention_masks, meta_data = self.preprocess(raw_data_col)
        dataset = TensorDataset(input_ids, attention_masks)
        dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)
        confidences_list = []
        labels_list = []
        for step, batch in enumerate(dataloader):
            in_ids, att_masks = batch
            with torch.no_grad():
                logits = self._model(in_ids, att_masks)[0]
            logits = F.softmax(logits, dim=2)
            confidences, labels = torch.max(logits, 2)
            confidences_list.extend(confidences.detach().cpu().numpy().tolist())
            labels_list.extend(labels.detach().cpu().numpy().tolist())
        infer_pdf = pd.DataFrame(meta_data.cpu()).astype(int)
        infer_pdf.columns = ["doc", "start", "stop"]
        infer_pdf["confidences"] = confidences_list
        infer_pdf["labels"] = labels_list
        infer_pdf["token_ids"] = input_ids.detach().cpu().numpy().tolist()

        del dataset
        del dataloader
        del logits
        del confidences
        del labels
        del confidences_list
        del labels_list
        parsed_df, confidence_df = self.__postprocess(infer_pdf)
        return parsed_df, confidence_df

    def __postprocess(self, infer_pdf):
        # cut overlapping edges
        infer_pdf["confidences"] = infer_pdf.apply(
            lambda row: row["confidences"][row["start"]:row["stop"]], axis=1
        )

        infer_pdf["labels"] = infer_pdf.apply(
            lambda row: row["labels"][row["start"]:row["stop"]], axis=1
        )

        infer_pdf["token_ids"] = infer_pdf.apply(
            lambda row: row["token_ids"][row["start"]:row["stop"]], axis=1
        )

        # aggregated logs
        infer_pdf = infer_pdf.groupby("doc").agg(
            {"token_ids": "sum", "confidences": "sum", "labels": "sum"}
        )

        # parse_by_label
        parsed_dfs = infer_pdf.apply(
            lambda row: self.__get_label_dicts(row), axis=1, result_type="expand"
        )
        parsed_df = pd.DataFrame(parsed_dfs[0].tolist())
        confidence_df = pd.DataFrame(parsed_dfs[1].tolist())
        if "X" in confidence_df.columns:
            confidence_df = confidence_df.drop(["X"], axis=1)
        confidence_df = confidence_df.applymap(np.mean)

        # decode cleanup
        parsed_df = self.__decode_cleanup(parsed_df)
        return parsed_df, confidence_df

    def __get_label_dicts(self, row):
        token_dict = defaultdict(str)
        confidence_dict = defaultdict(list)
        for label, confidence, token_id in zip(
            row["labels"], row["confidences"], row["token_ids"]
        ):
            text_token = self._vocab_lookup[token_id]
            if text_token[:2] != "##":
                # if not a subword use the current label, else use previous
                new_label = label
                new_confidence = confidence
            if self._label_map[new_label] in token_dict:
                token_dict[self._label_map[new_label]] = (
                    token_dict[self._label_map[new_label]] + " " + text_token
                )
            else:
                token_dict[self._label_map[new_label]] = text_token
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
