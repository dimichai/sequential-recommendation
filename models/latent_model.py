""" 
Created at 03/06/2019

@author: dimitris.michailidis
"""
import string
import torch
from torch import nn

from helpers.enums import LocationMode, LatentMode


class LatentGRU(nn.Module):
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
                 dropout_gru: float, use_cuda: bool, batch_size: int, latent_mode: LatentMode,
                 location_mode: LocationMode, city_size: int, distr_size: int,
                 context_size: int,
                 embedding_dim: int = -1):
        super(LatentGRU, self).__init__()
        self.input_size = input_size
        self.distr_size = distr_size
        self.city_size = city_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout_gru = dropout_gru
        self.dropout_input = dropout_input
        self.latent_mode = latent_mode
        self.location_mode = location_mode
        self.context_size = context_size
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        assert latent_mode != LatentMode.NONE
        assert location_mode != LocationMode.NONE

        # just the distance
        self.loc_input_size = 1

        if location_mode == LocationMode.DISTRICTHOT:
            self.distr_buffer = self.init_onehot_buffer(self.distr_size)
            self.loc_input_size = self.distr_size
        elif location_mode == LocationMode.CITYHOT:
            self.city_buffer = self.init_onehot_buffer(self.city_size)
            self.loc_input_size = self.city_size
        elif location_mode == LocationMode.FULLCONCAT:
            self.distr_buffer = self.init_onehot_buffer(self.distr_size)
            self.city_buffer = self.init_onehot_buffer(self.city_size)
            self.loc_input_size = city_size + distr_size + 1

        if latent_mode == LatentMode.LATENTAPPEND:
            # Mode1: Append location after hidden
            self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout_gru)
            self.h2o = nn.Linear(hidden_size + self.loc_input_size, output_size)
        elif latent_mode == LatentMode.LATENTEMBEDD:
            # Mode2: Context configuration
            assert context_size > 0, "Context size not set."
            assert embedding_dim > 0, "Embedding size not set."

            self.context_layer = nn.Linear(self.loc_input_size, context_size)
            self.context_activation = nn.ReLU()
            self.context_embedding_layer = nn.Linear(context_size, embedding_dim)
            self.id_embedding = nn.Embedding(input_size, self.embedding_dim)
            self.h2o = nn.Linear(hidden_size, output_size)
            self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_gru)
        elif latent_mode == LatentMode.LATENTMULTIPLY:
            # Mode3: Multiply location after hidden
            self.context_layer = nn.Linear(self.loc_input_size, hidden_size)
            self.context_activation = nn.Sigmoid()
            self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout_gru)
            self.h2o = nn.Linear(hidden_size, output_size)
        elif latent_mode == LatentMode.LATENTCONCAT:
            # Mode4: Concat location after hidden
            if self.context_size > -1:
                self.context_layer = nn.Linear(self.loc_input_size, context_size)
                self.h2o = nn.Linear(hidden_size + context_size, output_size)
            else:
                self.context_layer = nn.Linear(self.loc_input_size, hidden_size)
                self.h2o = nn.Linear(2 * hidden_size, output_size)

            self.context_activation = nn.ReLU()
            self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout_gru)

        # Universal
        self.final_act = nn.Tanh()
        self.onehot_buffer = self.init_onehot_buffer(self.input_size)
        # self.look_up = nn.Embedding(input_size, self.embedding_dim)
        # self.look_up = nn.Linear(input_size, self.embedding_dim)
        # self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_gru)

        self = self.to(self.device)

    def forward(self, batch_input, distance, hidden_state, city_input_idx=None, distr_input_idx=None):
        """
        Parameters:
        :param distance:
        :param batch_input: a batch of item indices from a session-parallel mini-batch.
        :param (B, H) hidden_state: the previous hidden state
        Returns:
        logit (B,C): Variable that stores the logits for the next items in the session-parallel mini-batch
        hidden: GRU hidden.
        """

        distance = distance.view(-1, 1)
        distance = distance.to(self.device)
        loc_input = distance

        if self.location_mode == LocationMode.DISTRICTHOT:
            loc_input = self.onehot_encode(distr_input_idx, self.distr_buffer)
        elif self.location_mode == LocationMode.CITYHOT:
            loc_input = self.onehot_encode(city_input_idx, self.city_buffer)
        elif self.location_mode == LocationMode.FULLCONCAT:
            distr_input = self.onehot_encode(distr_input_idx, self.distr_buffer)
            city_input = self.onehot_encode(city_input_idx, self.city_buffer)
            loc_input = torch.cat((distr_input, city_input), 1)
            loc_input = torch.cat((loc_input, distance), 1)

        if self.latent_mode == LatentMode.LATENTAPPEND:
            # Mode1: Append location after hidden
            id_encode = self.onehot_encode(batch_input, self.onehot_buffer)
            id_encode = id_encode.unsqueeze(0)

            output, hidden_state = self.gru(id_encode, hidden_state)
            output = output.view(-1, output.size(-1))
            output = torch.cat((output, loc_input), 1)
        elif self.latent_mode == LatentMode.LATENTEMBEDD:
            # Mode2: Context configuration
            id_embedding = batch_input.unsqueeze(0)
            id_embedding = self.id_embedding(id_embedding)

            latent_embedding = self.context_layer(loc_input)
            latent_embedding = self.context_activation(latent_embedding)
            latent_embedding = self.context_embedding_layer(latent_embedding)
            latent_embedding = latent_embedding.to(self.device)
            joint_embedding = id_embedding * latent_embedding
            joint_embedding = joint_embedding.to(self.device)

            output, hidden_state = self.gru(joint_embedding, hidden_state)
            output = output.view(-1, output.size(-1))
        elif self.latent_mode == LatentMode.LATENTMULTIPLY:
            # Mode3: Multiply location after hidden
            id_encode = self.onehot_encode(batch_input, self.onehot_buffer)
            id_encode = id_encode.unsqueeze(0)

            output, hidden_state = self.gru(id_encode, hidden_state)
            output = output.view(-1, output.size(-1))

            loc_embedding = self.context_layer(loc_input)
            loc_embedding = self.context_activation(loc_embedding)
            output = output * loc_embedding
        elif self.latent_mode == LatentMode.LATENTCONCAT:
            # Mode4: Concat location after hidden
            id_encode = self.onehot_encode(batch_input, self.onehot_buffer)
            id_encode = id_encode.unsqueeze(0)

            output, hidden_state = self.gru(id_encode, hidden_state)
            output = output.view(-1, output.size(-1))

            loc_embedding = self.context_layer(loc_input)
            loc_embedding = self.context_activation(loc_embedding)
            output = torch.cat((output, loc_embedding), 1)
        # elif self.latent_mode == LatentMode.LATENTPROJECT:
        #     # Mode5: Project location
        #     id_encode = self.onehot_encode(batch_input)
        #     id_encode = id_encode.unsqueeze(0)

        logit = self.final_act(self.h2o(output))

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

    def embedding_dropout(self, input):
        p_drop = torch.Tensor(input.size(0), 1).fill_(1 - self.dropout_input)  # (B,1)
        mask = torch.bernoulli(p_drop).expand_as(input) / (1 - self.dropout_input)  # (B,C)
        mask = mask.to(self.device)
        input = input * mask  # (B,C)

        return input
