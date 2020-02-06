import cudf
import torch
import logging
from clx.analytics import detector_utils as du
from torch.utils.dlpack import from_dlpack
from clx.analytics.detector import Detector
from clx.analytics.model.rnn_classifier import RNNClassifier

log = logging.getLogger(__name__)


class DGADetector(Detector):
    """
    This class provides multiple functionalities such as build, train and evaluate the RNNClassifier model 
    to distinguish legitimate and DGA domain names.
    """

    def init_model(self, char_vocab=128, hidden_size=100, n_domain_type=2, n_layers=3):
        """This function instantiates RNNClassifier model to train. And also optimizes to scale it and keep running on parallelism. 
        
        :param char_vocab: Vocabulary size is set to 128 ASCII characters.
        :type char_vocab: int
        :param hidden_size: Hidden size of the network.
        :type hidden_size: int
        :param n_domain_type: Number of domain types.
        :type n_domain_type: int
        :param n_layers: Number of network layers.
        :type n_layers: int
        """
        if self.model is None:
            model = RNNClassifier(char_vocab, hidden_size, n_domain_type, n_layers)
            self.leverage_model(model)

    def train_model(self, detector_dataset):
        """This function is used for training RNNClassifier model with a given training dataset. It returns total loss to determine model prediction accuracy.
        :param detector_dataset: Instance holds preprocessed data
        :type detector_dataset: DetectorDataset
        :return: Total loss
        :rtype: int

        Examples
        --------
        >>> from clx.analytics.dga_detector import DGADetector
        >>> partitioned_dfs = ... # partitioned_dfs = [df1, df2, ...] represents training dataset
        >>> dd = DGADetector()
        >>> dd.init_model()
        >>> dd.train_model(detector_dataset)
        1.5728906989097595
        """
        total_loss = 0
        i = 0
        for df in detector_dataset.partitioned_dfs:
            domains_len = df["type"].count()
            if domains_len > 0:
                types_tensor = self.__create_types_tensor(df["type"])
                df = df.drop(["type", "domain"])
                input, seq_lengths = self.__create_variables(df)
                model_result = self.model(input, seq_lengths)
                loss = self.__get_loss(model_result, types_tensor)
                total_loss += loss
                i = i + 1
                if i % 10 == 0:
                    print(
                        "[{}/{} ({:.0f}%)]\tLoss: {:.2f}".format(
                            i * domains_len,
                            detector_dataset.dataset_len,
                            100.0 * i * domains_len / detector_dataset.dataset_len,
                            total_loss / i * domains_len,
                        )
                    )
        return total_loss

    def predict(self, domains):
        """This function accepts cudf series of domains as an argument to classify domain names as benign/malicious and returns the learned label for each object in the form of cudf series.
        
        :param domains: List of domains.
        :type domains: cudf.Series
        :return: Predicted results with respect to given domains.
        :rtype: cudf.Series
        
        Examples
        --------
        >>> dd.predict(['nvidia.com', 'dgadomain'])
        0    0
        1    1
        Name: is_dga, dtype: int64
        """
        df = cudf.DataFrame({"domain": domains})
        domains_len = df["domain"].count()
        temp_df = du.str2ascii(df, domains_len)
        # Assigning sorted domains index to return learned labels as per the given input order.
        df.index = temp_df.index
        df["domain"] = temp_df["domain"]
        temp_df = temp_df.drop("domain")
        input, seq_lengths = self.__create_variables(temp_df)
        del temp_df
        model_result = self.model(input, seq_lengths)
        pred = model_result.data.max(1, keepdim=True)[1]
        type_ids = pred.view(-1).tolist()
        df["is_dga"] = type_ids
        df = df.sort_index()
        return df["is_dga"]

    def __create_types_tensor(self, type_series):
        """Create types tensor variable in the same order of sequence tensor"""
        types = type_series.to_array()
        types_tensor = torch.LongTensor(types)
        if torch.cuda.is_available():
            types_tensor = self.__set_var2cuda(types_tensor)
        return types_tensor

    def __create_variables(self, df):
        """
        Creates vectorized sequence for given domains and wraps around cuda for parallel processing.
        """
        seq_len_arr = df["len"].to_array()
        df = df.drop("len")
        seq_len_tensor = torch.LongTensor(seq_len_arr)
        seq_tensor = self.__df2tensor(df)
        # Return variables
        # DataParallel requires everything to be a Variable
        if torch.cuda.is_available():
            seq_tensor = self.__set_var2cuda(seq_tensor)
            seq_len_tensor = self.__set_var2cuda(seq_len_tensor)
        return seq_tensor, seq_len_tensor

    def __df2tensor(self, ascii_df):
        """
        Converts gdf -> dlpack tensor -> torch tensor
        """
        dlpack_ascii_tensor = ascii_df.to_dlpack()
        seq_tensor = from_dlpack(dlpack_ascii_tensor).long()
        return seq_tensor

    def evaluate_model(self, detector_dataset):
        """This function evaluates the trained model to verify it's accuracy.
        
        :param detector_dataset: Instance holds preprocessed data.
        :type detector_dataset: DetectorDataset
        :return: Model accuracy
        :rtype: decimal

        Examples
        --------
        >>> dd = DGADetector()
        >>> dd.init_model()
        >>> dd.evaluate_model(detector_dataset)
        Evaluating trained model ...
        Test set: Accuracy: 3/4 (0.75)
        """
        log.info("Evaluating trained model ...")
        correct = 0
        for df in detector_dataset.partitioned_dfs:
            target = self.__create_types_tensor(df["type"])
            df = df.drop(["type", "domain"])
            input, seq_lengths = self.__create_variables(df)
            output = self.model(input, seq_lengths)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        accuracy = float(correct) / detector_dataset.dataset_len
        print(
            "Test set: Accuracy: {}/{} ({})\n".format(
                correct, detector_dataset.dataset_len, accuracy
            )
        )
        return accuracy

    def __get_loss(self, model_result, types_tensor):
        loss = self.criterion(model_result, types_tensor)
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def __set_var2cuda(self, tensor):
        """
        Set variable to cuda.
        """
        return tensor.cuda()
