import torch
import torch.nn as nn
from rapidscyber.ml.provider.abstract_detector import AbstractDetector
from rapidscyber.ml.model.rnn_classifier import RNNClassifier


class DGADetector(AbstractDetector):
    def init_model(self, char_vocab=128, hidden_size=100, n_domain_type=2, n_layers=3):
        if self.model is None:
            if self.model is None:
                model = RNNClassifier(char_vocab, hidden_size, n_domain_type, n_layers)
                self.leverage_model(model)

    # Training
    def train_model(self, epoch, data_loader):
        total_loss = 0
        dataset_len = len(data_loader.dataset)
        for i, (domains, types) in enumerate(data_loader, 1):
            input, seq_lengths, target = self.__make_variables(
                domains, types, data_loader.dataset
            )
            output = self.model(input, seq_lengths)
            loss = self.__get_loss(output, target)
            total_loss += loss
            domains_len = len(domains)
            if i % 10 == 0:
                print(
                    "[{}] Epoch:  [{}/{} ({:.0f}%)]\tLoss: {:.2f}".format(
                        epoch,
                        i * domains_len,
                        len(data_loader.dataset),
                        100.0 * i * domains_len / dataset_len,
                        total_loss / i * domains_len,
                    )
                )
        return self.model, total_loss

    # Inference
    def predict(self, domains):
        input, seq_lengths, target = self.__make_variables(domains, [])
        output = self.model(input, seq_lengths)
        type_ids = self.__get_type_ids(output)
        return type_ids

    def __make_variables(self, domains, types, train_dataset=[]):
        sequence_and_length = [self.__str2ascii_arr(domain) for domain in domains]
        vectorized_seqs = [sl[0] for sl in sequence_and_length]
        seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
        return self.__pad_sequences(vectorized_seqs, seq_lengths, types, train_dataset)

    def __str2ascii_arr(self, msg):
        arr = [ord(c) for c in msg]
        return arr, len(arr)

    # pad sequences and sort the tensor
    def __pad_sequences(self, vectorized_seqs, seq_lengths, types, train_dataset):
        seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
        for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

        # Sort tensors by their length
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]

        # Also sort the target (types) in the same order
        target = self.__types2tensor(types, train_dataset)
        if len(types):
            target = target[perm_idx]

        # Return variables
        # DataParallel requires everything to be a Variable
        return (
            self.__create_variable(seq_tensor),
            self.__create_variable(seq_lengths),
            self.__create_variable(target),
        )

    def __types2tensor(self, types, train_dataset):
        type_ids = [train_dataset.get_type_id(type) for type in types]
        return torch.LongTensor(type_ids)

    def __get_loss(self, output, target):
        loss = self.criterion(output, target)
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def __get_type_ids(self, output):
        type_ids = []
        pred = output.data.max(1, keepdim=True)[1]
        for elm in pred.cpu().numpy():
            type_id = elm[0]
            if type_id == 1:
                type_ids.append(type_id)
            else:
                type_ids.append(0)
        return type_ids

    def __create_variable(self, tensor):
        # Do cuda() before wrapping with variable
        if torch.cuda.is_available():
            return tensor.cuda()
        else:
            return tensor
