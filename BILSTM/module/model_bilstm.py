import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MyLSTM(nn.Module):
    def __init__(self, dim_input, bilstm_input, hidden_dim, layer_size, dropout):
        super(MyLSTM, self).__init__()
        self.embedding = nn.Linear(dim_input, bilstm_input)
        self.bilstm = nn.LSTM(bilstm_input, hidden_dim, layer_size, batch_first=True, bidirectional=True, dropout=dropout)
        self.output = nn.Linear(hidden_dim * 2, out_features=2)

    def forward(self, input_tuple):

        seqs, lengths = input_tuple
        embedded = torch.tanh(self.embedding(seqs))
        seqs_packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True)
        seqs, _ = self.bilstm(seqs_packed)
        unpacked_output, _ = pad_packed_sequence(seqs, batch_first=True)
        _, (hidden, _) = self.bilstm(seqs_packed)
        last_output = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        output = self.output(last_output)

        return output