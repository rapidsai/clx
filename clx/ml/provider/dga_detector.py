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
This class provides multiple functionalities such as build, train and evaluate the RNNClassifier model 
to distinguish legitimate and DGA domain names.
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
            domains_len = df['domain'].count()
            if domains_len > 0:
                types_tensor = self.__create_types_tensor(df['type'])
                df = df.drop(['type', 'domain'])
                input, seq_lengths = self.__create_variables(df)
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
    This function accepts cudf series of domains as an argument to classify domain names as benign/malicious and 
    returns the learned label for each object in the form of cudf series.
    Example: 
        detector.predict(['nvidia.com', 'asfnfdjds']) = [1,0]
    """

    def predict(self, domains):
        df = cudf.DataFrame()
        df["domain"] = domains
        domains_len = df["domain"].count()
        temp_df = self.str2ascii(df, domains_len)
        # Assigning sorted domains index to return learned labels as per the given input order.
        df.index = temp_df.index
        df['domain'] = temp_df['domain']
        temp_df = temp_df.drop('domain')
        input, seq_lengths = self.__create_variables(temp_df)
        model_result = self.model(input, seq_lengths)
        pred = model_result.data.max(1, keepdim=True)[1]
        type_ids = pred.view(-1).tolist()
        df['is_dga'] = type_ids
        df = df.sort_index()
        return df['is_dga']

   
    def __create_types_tensor(self, type_series):
        """Create types tensor variable in the same order of sequence tensor"""
        types = type_series.to_array()
        types_tensor = torch.LongTensor(types)
        if torch.cuda.is_available():
            types_tensor = self.__set_var2cuda(types_tensor)
        return types_tensor

    def __create_variables(self, df):
        """Creates vectorized sequence for given domains and wraps around cuda for parallel processing."""
        seq_len_arr = df["len"].to_array()
        df  = df.drop('len')
        seq_len_tensor = torch.LongTensor(seq_len_arr)
        seq_tensor = self.__df2tensor(df)
        # Return variables
        # DataParallel requires everything to be a Variable
        if torch.cuda.is_available():
            seq_tensor = self.__set_var2cuda(seq_tensor)
            seq_len_tensor = self.__set_var2cuda(seq_len_tensor)
        return seq_tensor, seq_len_tensor

    def __df2tensor(self, ascii_df):
        """Converts gdf -> dlpack tensor -> torch tensor"""
        dlpack_ascii_tensor = ascii_df.to_dlpack()
        seq_tensor = from_dlpack(dlpack_ascii_tensor).long()
        return seq_tensor

    def str2ascii(self, df, domains_len):
        """This function sorts domain name entries in desc order based on the length of domain and converts domain name to ascii characters."""
        df["len"] = df["domain"].str.len()
        df = df.sort_values("len", ascending=False)
        splits = df["domain"].str.findall("[\w\.\-\@]")
        split_df = cudf.DataFrame()
        columns_cnt = len(splits)
        
        for i in range(0, columns_cnt):
            split_df[i] = splits[i]
        
        # https://github.com/rapidsai/cudf/issues/3123
        # Replace null's with ^.
        split_df = split_df.fillna("^")
        temp_df = cudf.DataFrame()
        for col in range(0, columns_cnt):
            ascii_darr = rmm.device_array(domains_len, dtype=np.int32)
            split_df[col].data.code_points(ascii_darr.device_ctypes_pointer.value)
            temp_df[col] = ascii_darr
        
        # https://github.com/rapidsai/cudf/issues/3123   
        # Replace ^ ascii value 94 with 0.
        temp_df = temp_df.replace(94, 0)
        temp_df['len']  = df['len']
        if 'type' in df.columns:
            temp_df['type'] = df['type']
        temp_df['domain']  = df['domain']
        temp_df.index = df.index
        return temp_df


    def evaluate_model(self, test_partitioned_dfs, dataset_len):
        """This function evaluates the trained model to verify it's accuracy."""
        print("Evaluating trained model ...")
        correct = 0
        for df in test_partitioned_dfs:
            target = self.__create_types_tensor(df['type'])
            df = df.drop(['type', 'domain'])
            input, seq_lengths  = self.__create_variables(df)
            output = self.model(input, seq_lengths)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        accuracy=float(correct)/dataset_len
        print('Test set: Accuracy: {}/{} ({})\n'.format(correct, dataset_len, accuracy))
        return accuracy

    def __get_loss(self, model_result, types_tensor):
        loss = self.criterion(model_result, types_tensor)
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def __set_var2cuda(self, tensor):
        """Set variable to cuda."""
        return tensor.cuda()
    