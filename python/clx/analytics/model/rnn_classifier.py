# Original code at https://github.com/spro/practical-pytorch
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

DROPOUT = 0.0


class RNNClassifier(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, n_layers, bidirectional=True
    ):
        super(RNNClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            dropout=DROPOUT,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, seq_lengths):
        # Note: we run this all at once (over the whole input sequence)
        # input shape: B x S (input size)
        # transpose to make S(sequence) x B (batch)
        input = input.t()
        batch_size = input.size(1)

        # Make a hidden
        hidden = self._init_hidden(batch_size)

        # Embedding S x B -> S x B x I (embedding size)
        embedded = self.embedding(input)

        # Pack them up nicely
        gru_input = pack_padded_sequence(embedded, seq_lengths.data.cpu().numpy())

        # To compact weights again call flatten_parameters().
        self.gru.flatten_parameters()
        output, hidden = self.gru(gru_input, hidden)
        # output = self.dropout(output)

        # Use the last layer output as FC's input
        # No need to unpack, since we are going to use hidden
        fc_output = self.fc(hidden[-1])
        return fc_output

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(
            self.n_layers * self.n_directions, batch_size, self.hidden_size
        )
        # creating variable
        if torch.cuda.is_available():
            return hidden.cuda()
        else:
            return hidden
