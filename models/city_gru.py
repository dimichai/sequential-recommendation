""" 
Created at 22/05/2019

@author: dimitris.michailidis
GRU that takes as input a one-hot vector of the city an item is located at.
"""

import string

import torch
from torch import nn

from helpers.enums import ParallelMode, LocationMode


class CityGRU(nn.Module):
    city_size: int
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

    def __init__(self, city_size: int, distr_size: int, output_size: int, hidden_size: int, num_layers: int,
                 dropout_input: float, dropout_gru: float, use_cuda: bool, batch_size: int,
                 parallel_mode=ParallelMode.NONE, location_mode=LocationMode.CITYHOT):
        super(CityGRU, self).__init__()
        self.city_size = city_size
        self.distr_size = distr_size
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
        # In encoder-parallel mode, we don't care about the final layer, we need only the encoder level outputs
        if parallel_mode == ParallelMode.NONE or parallel_mode == ParallelMode.DECODER:
            self.h2o = nn.Linear(hidden_size, output_size)

        self.city_buffer = self.init_onehot_buffer(self.city_size)
        if location_mode == LocationMode.DISTRICTHOT:
            self.distr_buffer = self.init_onehot_buffer(self.distr_size)
            self.city_size = self.distr_size
        elif location_mode == LocationMode.FULLCONCAT:
            self.distr_buffer = self.init_onehot_buffer(self.distr_size)
            self.city_size = city_size + distr_size + 1  # one for the distance

        self.gru = nn.GRU(self.city_size, self.hidden_size, self.num_layers, dropout=self.dropout_gru)

        self = self.to(self.device)

    def forward(self, city_input_idx, hidden_state, distr_input_idx=None, distance=None):
        """
        Args:
            :param city_input_idx: a batch of city indices from a session-parallel mini-batch.
            :param hidden_state: previous hidden state of the network
        Returns:
            logit (B,C): Variable that stores the logits for the next items in the session-parallel mini-batch
            hidden: GRU hidden state
        """
        if self.location_mode == LocationMode.DISTRICTHOT:
            input_embedding = self.onehot_encode(city_input_idx, self.distr_buffer)
        else:
            input_embedding = self.onehot_encode(city_input_idx, self.city_buffer)

        input_embedding = input_embedding.unsqueeze(0)

        if distr_input_idx is not None:
            distr_embedding = self.onehot_encode(distr_input_idx, self.distr_buffer)
            distr_embedding = distr_embedding.unsqueeze(0)
            input_embedding = torch.cat((input_embedding, distr_embedding), 2)

        if distance is not None:
            distance = distance.view(1, -1, 1)
            distance = distance.to(self.device)
            input_embedding = torch.cat((input_embedding, distance), 2)

        # the batch input encoding and the previous hidden state
        output, hidden_state = self.gru(input_embedding, hidden_state)
        output = output.view(-1, output.size(-1))
        logit = torch.Tensor()

        if self.parallel_mode == ParallelMode.NONE:
            # hidden to output through a feedforward layer and then through the final activation
            logit = self.final_act(self.h2o(output))
        elif self.parallel_mode == ParallelMode.DECODER:
            logit = self.h2o(output)

        return logit, hidden_state, output

    def init_onehot_buffer(self, input_size: int):
        """
        Initializes buffer for storing the minibatch-input distances.
        :return: FloatTensor the buffer.
        """
        onehot_buffer = torch.FloatTensor(self.batch_size, input_size)
        onehot_buffer = onehot_buffer.to(self.device)

        return onehot_buffer

    def onehot_encode(self, input, buffer):
        """
        Returns a one-hot vector corresponding to the input

        Args:
            input (B,): torch.LongTensor of item indices
            :param buffer: the buffer to store the encode
        Returns:
            onehot (B,C): torch.FloatTensor of one-hot vectors
        """

        buffer.zero_()
        index = input.view(-1, 1)
        onehot = buffer.scatter_(1, index, 1)

        return onehot

    def init_hidden(self):
        """
        Initializes the hidden state of the GRUnit
        """
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        return h0
