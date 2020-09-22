""" 
Created at 17/05/2019

@author: dimitris.michailidis
"""

import string

import torch
from torch import nn

from helpers.enums import CombinationMode, ParallelMode


class ParallelModel(nn.Module):
    input_size: int
    output_size: int
    # Hidden to output layer - feedforward
    h2o: nn.Linear
    final_act: classmethod
    device: string

    def __init__(self, input_size: int, output_size: int, use_cuda: bool, parallel_mode: string,
                 combination_mode=CombinationMode.CONCAT):
        super(ParallelModel, self).__init__()

        self.final_act = nn.Tanh()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.parallel_mode = parallel_mode
        self.combination_mode = combination_mode
        self.h2o = nn.Linear(input_size, output_size)
        if combination_mode == CombinationMode.CONCAT:
            self.h2o = nn.Linear(2 * input_size, output_size)

        self = self.to(self.device)

    def forward(self, argument1, argument2):
        """
        Args:
            argument1 (B, h): output of the first parallel network
            argument2 (B, h): output of the second parallel network
        Returns:
            logit (B,C): Variable that stores the logits for the next items in the session-parallel mini-batch
            hidden: GRU hidden state
        """
        # todo convert to argument
        a = 0.9
        output = torch.Tensor()
        if self.combination_mode == CombinationMode.CONCAT:
            output = torch.cat((argument1, argument2), 1)
        elif self.combination_mode == CombinationMode.ADD:
            output = argument1 + argument2
        elif self.combination_mode == CombinationMode.MULTIPLY:
            output = argument1 * argument2
        elif self.combination_mode == CombinationMode.WEIGHTED_SUM:
            output = a * argument1 + (1 - a) * argument2

        if self.parallel_mode == ParallelMode.ENCODER:
            # hidden to output through a feedforward layer and then through the final activation
            logit = self.final_act(self.h2o(output))
        elif self.parallel_mode == ParallelMode.DECODER:
            logit = self.final_act(output)
        elif self.parallel_mode == ParallelMode.HIDDEN:
            output = output.squeeze(0)
            logit = self.final_act(self.h2o(output))

        return logit
