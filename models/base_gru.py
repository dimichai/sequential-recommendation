"""
Created at 05/04/2019

@author: dimitris.michailidis
"""
import csv
import string

import torch
from torch import nn

from helpers.enums import ParallelMode, LocationMode


class SimpleGRU(nn.Module):
    input_size: int
    output_size: int
    hidden_size: int
    num_layers: int
    batch_size: int
    dropout_gru: float
    dropout_input: float
    # Hidden to output layer - feedforward
    h2o: nn.Linear
    final_act: classmethod
    gru: nn.GRU
    device: string
    embedding_dim: int

    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int, dropout_input: float,
                 dropout_gru: float, use_cuda: bool, batch_size: int, embedding_dim: int = -1,
                 parallel_mode=ParallelMode.NONE, location_mode=LocationMode.NONE):
        super(SimpleGRU, self).__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout_gru = dropout_gru
        self.dropout_input = dropout_input
        self.final_act = nn.Tanh()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.parallel_mode = parallel_mode
        self.location_mode = location_mode
        # In encoder-level parallel mode, we don't care about the final layer, we need only the encoder level outputs
        if parallel_mode == ParallelMode.NONE or parallel_mode == ParallelMode.DECODER:
            self.h2o = nn.Linear(hidden_size, output_size)

        if location_mode == LocationMode.CONCAT:
            self.input_size = self.input_size + 1

        self.onehot_buffer = self.initialize_onehot_buffer()

        if self.embedding_dim != -1:
            self.look_up = nn.Embedding(input_size, self.embedding_dim)
            self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_gru)
        else:
            self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout_gru)
        self = self.to(self.device)

    def forward(self, batch_input, hidden_state, distance=None):
        """
        Parameters:
        :param batch_input: a batch of item indices from a session-parallel mini-batch.
        :param (B, H) hidden_state: the previous hidden state
        Returns:
        logit (B,C): Variable that stores the logits for the next items in the session-parallel mini-batch
        hidden: GRU hidden.
        """
        if self.embedding_dim == -1:
            embedded = self.onehot_encode(batch_input)
            if self.training and self.dropout_input > 0:
                embedded = self.embedding_dropout(embedded)
            embedded = embedded.unsqueeze(0)
        else:
            embedded = batch_input.unsqueeze(0)
            embedded = self.look_up(embedded)

        if distance is not None:
            distance = distance.view(1, -1, 1)
            distance = distance.to(self.device)
            embedded = torch.cat((embedded, distance), 2)

        # the batch input encoding and the previous hidden state
        output, hidden_state = self.gru(embedded, hidden_state)
        output = output.view(-1, output.size(-1))
        logit = torch.Tensor()

        LOG_FILE = './logit_logs/' + 'base_logits' + '.csv'

        if self.parallel_mode == ParallelMode.NONE:
            # output = self.h2o(output)
            # ranking = self.final_act(output)
            # hidden to output through a feedforward layer and then through the final activation
            # with open(LOG_FILE, 'a+', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow(output)
            #     writer.writerow(ranking)
            logit = self.final_act(self.h2o(output))
        elif self.parallel_mode == ParallelMode.DECODER:
            logit = self.h2o(output)

        return logit, hidden_state, output

    def initialize_onehot_buffer(self):
        """
        Initializes 1-of-N buffer for storing the minibatch-input.
        :return: FloatTensor the buffer.
        """
        onehot_buffer = torch.FloatTensor(self.batch_size, self.output_size)
        onehot_buffer = onehot_buffer.to(self.device)

        return onehot_buffer

    def onehot_encode(self, input):
        """
        Returns a one-hot vector corresponding to the input

        Args:
            input (B,): torch.LongTensor of item indices
        Returns:
            onehot (B,C): torch.FloatTensor of one-hot vectors
        """

        self.onehot_buffer.zero_()
        index = input.view(-1, 1)
        onehot = self.onehot_buffer.scatter_(1, index, 1)

        return onehot

    def init_hidden(self):
        """
        Initializes the hidden state of the GRUnit
        """
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        return h0

    def embedding_dropout(self, input):
        p_drop = torch.Tensor(input.size(0), 1).fill_(1 - self.dropout_input)  # (B,1)
        mask = torch.bernoulli(p_drop).expand_as(input) / (1 - self.dropout_input)  # (B,C)
        mask = mask.to(self.device)
        input = input * mask  # (B,C)

        return input
