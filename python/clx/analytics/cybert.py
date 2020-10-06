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

import cupy
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.dlpack import from_dlpack
from transformers import BertForTokenClassification

log = logging.getLogger(__name__)


class Cybert:
    """
    Cyber log parsing using BERT. This class provides methods for
    loading models, prediction, and postprocessing.
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

    def load_model(
        self, model_filepath, config_filepath, pretrained_model="bert-base-cased"
    ):
        """
        Load cybert model.

        :param model_filepath: Filepath of the model (.pth or .bin) to
        be loaded
        :type model_filepath: str
        :param label_map_filepath: Config file (.json) to be
        used
        :type label_map_filepath: str
        :param pretrained_model: Name of pretrained model to be loaded from
        transformers
        repo, default is bert-base-cased
        :type pretrained_model: str

        Examples
        --------
        >>> from clx.analytics.cybert import Cybert
        >>> cyparse = Cybert()
        >>> cyparse.load_model('/path/to/model.pth', '/path/to/config.json')
        """
        with open(config_filepath) as f:
            config = json.load(f)
        self._label_map = {int(k): v for k, v in config["id2label"].items()}
        model_state_dict = torch.load(model_filepath)
        self._model = BertForTokenClassification.from_pretrained(
            pretrained_model,
            state_dict=model_state_dict,
            num_labels=len(self._label_map),
        )
        self._model.cuda()
        self._model.eval()

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

        byte_count = raw_data_col.str.byte_count()
        max_num_chars = byte_count.sum()
        max_rows_tensor = int((byte_count / 120).ceil().sum())

        input_ids, att_mask, meta_data = raw_data_col.str.subword_tokenize(
            self._hashpath,
            128,
            116,
            max_num_strings=len(raw_data_col),
            max_num_chars=max_num_chars,
            max_rows_tensor=max_rows_tensor,
            do_lower=False,
            do_truncate=False,
        )

        num_rows = int(len(input_ids) / 128)
        input_ids = from_dlpack(
            (input_ids.reshape(num_rows, 128).astype(cupy.float)).toDlpack()
        )
        att_mask = from_dlpack(
            (att_mask.reshape(num_rows, 128).astype(cupy.float)).toDlpack()
        )
        meta_data = meta_data.reshape(num_rows, 3)

        return input_ids.type(torch.long), att_mask.type(torch.long), meta_data

    def inference(self, raw_data_col):
        """
        Cybert inference and postprocessing on dataset

        :param raw_data_col: logs to be processed
        :type raw_data_col: cudf.Series
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
        dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=32)
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
        infer_pdf = pd.DataFrame(meta_data).astype(int)
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
            lambda row: self.__get_label_dicts(row), axis=1, result_type="expand"
        )
        parsed_df = pd.DataFrame(parsed_dfs[0].tolist())
        confidence_df = pd.DataFrame(parsed_dfs[1].tolist())
        confidence_df = confidence_df.drop(["X"], axis=1).applymap(np.mean)

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
