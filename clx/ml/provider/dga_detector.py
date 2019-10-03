import cudf
import torch
import logging
import numpy as np
from librmm_cffi import librmm as rmm
from torch.utils.dlpack import from_dlpack
from clx.ml.provider.detector import Detector
from clx.ml.model.rnn_classifier import RNNClassifier

log = logging.getLogger(__name__)

"""
This class provides multiple functionalities to build and train the RNNClassifier model 
to distinguish between the legitimate and DGA domain names.
"""


class DGADetector(Detector):
    """
    This function instantiates RNNClassifier model to train. And also optimizes to scale it 
    and keep running on parallelism. 
    """

    def init_model(self, char_vocab=128, hidden_size=100, n_domain_type=2, n_layers=3):
        if self.model is None:
            model = RNNClassifier(char_vocab, hidden_size, n_domain_type, n_layers)
            self.leverage_model(model)

    """
    This function is used for training RNNClassifier model with a given training dataset. 
    It returns total loss to determine model prediction accuracy.
    """

    def train_model(self, partitioned_dfs, dataset_len):
        total_loss = 0
        i = 0
        for df in partitioned_dfs:
            domains_len = df["domain"].count()
            input, seq_lengths, df = self.__create_variables(df, domains_len)
            types_tensor = self.__create_types_tensor(df)
            model_result = self.model(input, seq_lengths)
            loss = self.__get_loss(model_result, types_tensor)
            total_loss += loss
            i = i + 1
            if i % 10 == 0:
                print(
                    "[{}/{} ({:.0f}%)]\tLoss: {:.2f}".format(
                        i * domains_len,
                        dataset_len,
                        100.0 * i * domains_len / dataset_len,
                        total_loss / i * domains_len,
                    )
                )
        return total_loss

    """
    This function accepts array of domains as an argument to classify domain names as benign/malicious and 
    returns the learned label for each object in the array.
    Example: 
        detector.predict(['nvidia.com', 'asfnfdjds']) = [1,0]
    """

    def predict(self, domains):
        df = cudf.DataFrame()
        df["domain"] = domains
        domains_len = df["domain"].count()
        ascii_df, seq_lengths, df = self.__create_variables(df, domains_len)
        model_result = self.model(ascii_df, seq_lengths)
        type_ids = self.__get_type_ids(model_result)
        return type_ids

    # Create types tensor variable in the same order of sequence tensor
    def __create_types_tensor(self, df):
        if "type" in df:
            types = df["type"].to_array()
            types_tensor = torch.LongTensor(types)
            if torch.cuda.is_available():
                types_tensor = self.__set_var2cuda(types_tensor)
            return types_tensor

    # Creates vectorized sequence for given domains and wraps around cuda for parallel processing.
    def __create_variables(self, df, domains_len):
        ascii_df, df = self.__str2ascii(df, domains_len)
        seq_tensor = self.__df2tensor(ascii_df)
        seq_len_arr = df["len"].to_array()
        seq_len_tensor = torch.LongTensor(seq_len_arr)
        # Return variables
        # DataParallel requires everything to be a Variable
        if torch.cuda.is_available():
            seq_tensor = self.__set_var2cuda(seq_tensor)
            seq_len_tensor = self.__set_var2cuda(seq_len_tensor)
        return seq_tensor, seq_len_tensor, df

    def __df2tensor(self, ascii_df):
        dlpack_ascii_tensor = ascii_df.to_dlpack()
        seq_tensor = from_dlpack(dlpack_ascii_tensor).long()
        return seq_tensor

    def __str2ascii(self, df, domains_len):
        df["len"] = df["domain"].str.len()
        df = df.sort_values("len", ascending=False)
        splits = df["domain"].str.findall("[\w\.\-\@]")
        split_df = cudf.DataFrame()
        columns_cnt = len(splits)
        for i in range(0, columns_cnt):
            split_df[i] = splits[i]
        # Replace null's with ^.
        split_df = split_df.fillna("^")
        ascii_df = cudf.DataFrame()
        for col in range(0, columns_cnt):
            ascii_darr = rmm.device_array(domains_len, dtype=np.int32)
            split_df[col].data.code_points(ascii_darr.device_ctypes_pointer.value)
            ascii_df[col] = ascii_darr
        # Replace ^ ascii value 94 with 0.
        ascii_df = ascii_df.replace(94, 0)
        return ascii_df, df

    def __get_loss(self, model_result, types_tensor):
        loss = self.criterion(model_result, types_tensor)
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # List of type id's from GPU memory to CPU.
    def __get_type_ids(self, model_result):
        type_ids = []
        pred = model_result.data.max(1, keepdim=True)[1]
        for elm in pred.cpu().numpy():
            type_id = elm[0]
            if type_id == 1:
                type_ids.append(type_id)
            else:
                type_ids.append(0)
        return type_ids

    # Set variable to cuda.
    def __set_var2cuda(self, tensor):
        return tensor.cuda()
