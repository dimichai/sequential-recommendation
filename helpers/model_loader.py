""" 
Created at 11/04/2019

@author: dimitris.michailidis
To load and saved models from checkpoints, etc.
"""
import datetime
import os
import string

import torch

from experiments.olx_clicks.args import Args


class ModelLoader:
    checkpoint_dir: string
    model_dir: string
    args: Args
    model_store_loc: string  # where to store the model 'gpu' / 'cpu'

    def __init__(self, args: Args):
        """
        Initializes the model_load to save and load checkpoints of the model
        :param args:
            the arguments of the model - saves them in a file in checkpoint.
        """
        self.checkpoint_dir = args.checkpoint_dir
        self.args = args
        self.model_store_loc = 'cuda' if args.use_cuda else 'cpu'

        self.__create_checkpoint_dir()

    def __create_checkpoint_dir(self):
        """
        Creates the checkpoint directory if it does not exist.
        :return:
        """
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def initialize_model_checkpoint(self):
        """
        Initializes the current moel checkpoint directory
        :return:
        """
        print("PARAMETERS" + "-" * 10)
        now = datetime.datetime.now()
        model_dir = '{:02d}{:02d}{:02d}{:02d}'.format(now.month, now.day, now.hour, now.minute)
        path = os.path.join(self.checkpoint_dir, model_dir)

        if not os.path.exists(path):
            os.mkdir(path)

        self.model_dir = path

        with open(os.path.join(self.model_dir, 'parameter.txt'), 'w') as f:
            for attr, value in sorted(self.args.__dict__.items()):
                print("{}={}".format(attr.upper(), value))
                f.write("{}={}\n".format(attr.upper(), value))

        print("---------" + "-" * 10)

    def load_pretrained_model(self, path: string):
        """
        Loads pretrained model from given path.
        :return:
        """
        print("Loading pre trained model from {}".format(path))
        checkpoint = torch.load(path, map_location=self.model_store_loc)
        model = checkpoint["model"]
        return model

    def save_model_checkpoint(self, checkpoint):
        model_name = os.path.join(self.model_dir, "model_{0:05d}.pt".format(checkpoint['epoch']))
        torch.save(checkpoint, model_name)
        print("Save model as %s" % model_name)
