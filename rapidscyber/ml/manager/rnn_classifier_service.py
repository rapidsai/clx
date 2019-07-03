import time
import torch
import logging
from rapidscyber.ml.model.rnn_classifier import RNNClassifier

log = logging.getLogger("RNNClassifierService")


class RNNClassifierService:
    def __init__(self, model, optimizer, criterion, train_loader=None, b_size=64):
        self.__classifier = model
        self.train_loader = train_loader
        self.batch_size = b_size
        self.__criterion = criterion
        self.__optimizer = optimizer

    @property
    def classifier(self):
        return self.__classifier
    
    @property
    def optimizer(self):
        return self.__optimizer

    @property
    def criterion(self):
        return self.__criterion

    def get_item(self, output, target):
        loss = self.criterion(output, target)
        self.classifier.zero_grad()
        loss.backward()
        self.optimizer.step() 
        return loss.item()

    # Training
    def train_model(self, epoch):
        total_loss = 0
        train_loader_len = len(self.train_loader.dataset)
        for i, (domains, types) in enumerate(self.train_loader, 1):
            input, seq_lengths, target = self.make_variables(domains, types)
            output = self.classifier(input, seq_lengths)
            loss = self.get_item(output, target) 
            total_loss += loss
            domains_len = len(domains)
            if i % 10 == 0:
                print(
                    "[{}] Epoch:  [{}/{} ({:.0f}%)]\tLoss: {:.2f}".format(
                        epoch,
                        i * domains_len,
                        len(self.train_loader.dataset),
                        100.0 * i * domains_len / train_loader_len,
                        total_loss / i * domains_len,
                    )
                )
        return self.classifier, total_loss

    def get_type_ids(self, output):
        type_ids = []
        pred = output.data.max(1, keepdim=True)[1]
        for elm in pred.cpu().numpy():
            type_id = elm[0]
            if type_id == 1:
                type_ids.append(type_id)
            else:
                type_ids.append(0)
        return type_ids

    # Inference
    def predict(self, domains):
        input, seq_lengths, target = self.make_variables(domains, [])
        output = self.classifier(input, seq_lengths)
        type_ids = self.get_type_ids(output)
        return type_ids

    def make_variables(self, domains, types):
        sequence_and_length = [self.str2ascii_arr(domain) for domain in domains]
        vectorized_seqs = [sl[0] for sl in sequence_and_length]
        seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
        return self.pad_sequences(vectorized_seqs, seq_lengths, types)

    def create_variable(self, tensor):
        # Do cuda() before wrapping with variable
        if torch.cuda.is_available():
            return tensor.cuda()
        else:
            return tensor

    # pad sequences and sort the tensor
    def pad_sequences(self, vectorized_seqs, seq_lengths, types):
        seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
        for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

        # Sort tensors by their length
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]

        # Also sort the target (types) in the same order
        target = self.types2tensor(types)
        if len(types):
            target = target[perm_idx]

        # Return variables
        # DataParallel requires everything to be a Variable
        return (
            self.create_variable(seq_tensor),
            self.create_variable(seq_lengths),
            self.create_variable(target),
        )

    def types2tensor(self, types):
        type_ids = [self.train_loader.dataset.get_type_id(type) for type in types]
        return torch.LongTensor(type_ids)

    def str2ascii_arr(self, msg):
        arr = [ord(c) for c in msg]
        return arr, len(arr)

