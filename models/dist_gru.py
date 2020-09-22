""" 
Created at 08/05/2019

@author: dimitris.michailidis
Distance based GRU. The input is a vector of size n_items with the distances to this item.
"""
import string

import torch
from torch import nn

from helpers.enums import ParallelMode, LocationMode


class DistanceGRU(nn.Module):
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

    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int, dropout_input: float,
                 dropout_gru: float, use_cuda: bool, batch_size: int, loc_mode=LocationMode.DISTHOT,
                 parallel_mode=ParallelMode.NONE):
        super(DistanceGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout_gru = dropout_gru
        self.dropout_input = dropout_input
        self.location_mode = loc_mode
        self.final_act = nn.Tanh()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.parallel_mode = parallel_mode
        # In parallel mode, we don't care about the final layer, we need only the encoder level outputs
        if parallel_mode == ParallelMode.NONE or parallel_mode == ParallelMode.DECODER:
            self.h2o = nn.Linear(hidden_size, output_size)

        if self.location_mode == LocationMode.DISTHOT:
            self.dist_buffer = self.init_distance_buffer()
        elif self.location_mode == LocationMode.DISTANCE:
            self.input_size = 1

        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout_gru)

        self = self.to(self.device)

    def forward(self, batch_input, distance, hidden_state):
        """
        Args:
            batch_input (B,): a batch of item indices from a session-parallel mini-batch.
        Returns:
            logit (B,C): Variable that stores the logits for the next items in the session-parallel mini-batch
            hidden: GRU hidden state
        """
        if self.location_mode == LocationMode.DISTHOT:
            embedded = self.distance_encode(batch_input, distance)
        elif self.location_mode == LocationMode.DISTANCE:
            embedded = distance.view(-1, 1)
            embedded = embedded.to(self.device)
        embedded = embedded.unsqueeze(0)

        # the batch input encoding and the previous hidden state
        output, hidden_state = self.gru(embedded, hidden_state)
        output = output.view(-1, output.size(-1))
        logit = torch.Tensor()

        if self.parallel_mode == ParallelMode.NONE:
            # hidden to output through a feedforward layer and then through the final activation
            logit = self.final_act(self.h2o(output))
        elif self.parallel_mode == ParallelMode.DECODER:
            logit = self.h2o(output)

        return logit, hidden_state, output

    def init_distance_buffer(self):
        """
        Initializes buffer for storing the minibatch-input distances.
        :return: FloatTensor the buffer.
        """
        dist_buffer = torch.FloatTensor(self.batch_size, self.output_size)
        dist_buffer = dist_buffer.to(self.device)

        return dist_buffer

    def distance_encode(self, input, distance):
        """
        Returns a one-hot vector corresponding to the input

        Args:
            input (B,): torch.LongTensor of item indices
        Returns:
            onehot (B,C): torch.FloatTensor of one-hot vectors
        """

        self.dist_buffer.zero_()
        index = input.view(-1, 1)
        distance = distance.view(-1, 1)
        distance = distance.to(self.device)
        onehot = self.dist_buffer.scatter_(1, index, distance)

        return onehot

    def init_hidden(self):
        """
        Initializes the hidden state of the GRUnit
        """
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        return h0
