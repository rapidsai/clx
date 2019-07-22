import torch
from rapidscyber.ml.provider.detector import Detector
from rapidscyber.ml.model.rnn_classifier import RNNClassifier

"""
This class provides multiple functionalities to build and train the RNNClassifier model 
to distinguish between the legitimate and DGA domain names.
"""


class DGADetector(Detector):
    """
    This function instantiates a RNNClassifier model to train and also optimizes to scale it 
    and keep running on parallelism. 
    """
    def init_model(self, char_vocab=128, hidden_size=100, n_domain_type=2, n_layers=3):
        if self.model is None:
            model = RNNClassifier(char_vocab, hidden_size, n_domain_type, n_layers)
            self.leverage_model(model)

    """
    This function is used for training RNNClassifier model with given training dataset 
    and return total loss to determine accuracy.
    """
    def train_model(self, data_loader):
        total_loss = 0
        dataset_len = len(data_loader.dataset)
        for i, (domains, types) in enumerate(data_loader, 1):
            input, seq_lengths, perm_idx = self.__create_variables(domains)
            types_tensor = self.__create_types_tensor(perm_idx, types=types, train_dataset=data_loader.dataset)
            model_result = self.model(input, seq_lengths)
            loss = self.__get_loss(model_result, types_tensor)
            total_loss += loss
            domains_len = len(domains)
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
    This function accepts one argument(array of domains) to classify domain names as benign/malicious and 
    returns the learned label for each object in the array.
    Example: 
        detector.predict(['nvidia.com', 'asfnfdjds']) = [1,0]
    """
    def predict(self, domains):
        input, seq_lengths, perm_idx = self.__create_variables(domains)
        model_result = self.model(input, seq_lengths)
        type_ids = self.__get_type_ids(model_result)
        return type_ids

    # Creates vectorized sequence for given domains and wraps around cuda for parallel processing.
    def __create_variables(self, domains):
        vectorized_seqs, seq_lengths = self.__vectorize_seqs(domains)
        seq_tensor, seq_lengths, perm_idx = self.__pad_sequences(
            vectorized_seqs, seq_lengths
        )
        return seq_tensor, seq_lengths, perm_idx
    
    # Create types tensor variable and sorts in the same order of sequence tensor
    def __create_types_tensor(self, perm_idx, types=None, train_dataset=None):
        if (types and train_dataset) is not None:
            types_tensor = self.__types2tensor(types, train_dataset)
            if len(types):
                types_tensor = types_tensor[perm_idx]
            if torch.cuda.is_available():
                types_tensor = self.__set_var2cuda(types_tensor)
            return types_tensor

    def __vectorize_seqs(self, domains):
        sequence_and_length = [self.__str2ascii_arr(domain) for domain in domains]
        vectorized_seqs = [sl[0] for sl in sequence_and_length]
        seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
        return vectorized_seqs, seq_lengths

    
    def __pad_sequences(self, vectorized_seqs, seq_lengths):
        seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
        for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
        # Sort tensors by their length
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        # Return variables
        # DataParallel requires everything to be a Variable
        if torch.cuda.is_available():
            seq_tensor = self.__set_var2cuda(seq_tensor)
            seq_lengths = self.__set_var2cuda(seq_lengths)
            
        return seq_tensor, seq_lengths, perm_idx

    def __str2ascii_arr(self, msg):
        arr = [ord(c) for c in msg]
        return arr, len(arr)

    def __types2tensor(self, types, train_dataset):
        type_ids = [train_dataset.get_type_id(type) for type in types]
        return torch.LongTensor(type_ids)

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
        
