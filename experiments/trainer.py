""" 
Created at 08/04/2019

@author: dimitris.michailidis
"""
import time

import numpy as np
import torch

from experiments.olx_clicks.args import Args
from helpers.dataset import Dataset
from helpers.distance_calculator import calculate_session_item_distance
from helpers.enums import LocationMode, LatentMode
from helpers.model_loader import ModelLoader
from helpers.session_parallel_loader import SessionParallelLoader
from models.evaluation import Evaluation
from models.lossfunction import LossFunction
from models.optimizer import Optimizer


def reset_hidden(hidden, session_mask):
    """
    Resets the hidden state when a session terminates.
    :param hidden: the hidden state
    :param session_mask: the mask of the terminated sessions
    """
    if len(session_mask) != 0:
        hidden[:, session_mask, :] = 0
    return hidden


class Trainer:
    start_time: float

    def __init__(self, model, train_data: Dataset, eval_data: Dataset, optim: Optimizer, use_cuda: bool,
                 loss_func: LossFunction, args: Args, model_loader: ModelLoader = None):
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.optim = optim
        self.loss_func = loss_func
        self.evaluation = Evaluation(self.model, self.loss_func, use_cuda, batch_size=args.batch_size,
                                     location_mode=args.location_mode, k=args.topk, latent_mode=args.latent_mode)
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.args = args
        self.model_loader = model_loader
        self.location_mode = args.location_mode

    def train(self, start_epoch: int, end_epoch: int, start_time: float = None, random_search=False):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        train_loss = 0
        loss = 0
        recall = 0
        mrr = 0
        for epoch in range(start_epoch, end_epoch + 1):
            st = time.time()
            # Loss while training
            train_loss = self.train_epoch()
            loss, recall, mrr, recall5, mrr5, recall10, mrr10 = self.evaluation.eval(self.eval_data)

            print(f'Epoch: {epoch}, train-loss: {train_loss:.05}, test-loss: {loss:.05}, recall@{self.evaluation.topk}: {recall:.05}, '
                  f'mrr@{self.evaluation.topk}: {mrr:.05}, recall@10: {recall10:.05}, mrr@10: {mrr10:.05}, recall@5: {recall5:.05}, mrr@5: {mrr5:.05}, '
                  f'time: {time.time() - st}')

            checkpoint = {
                'model': self.model,
                'args': self.args,
                'epoch': epoch,
                'optim': self.optim,
                'loss': loss,
                'recall': recall,
                'mrr': mrr
            }

            if self.model_loader:
                self.model_loader.save_model_checkpoint(checkpoint)

        if random_search:
            return train_loss, loss, recall, mrr

    def train_epoch(self):
        # set the model in train mode
        self.model.train()

        losses = []

        hidden = self.model.init_hidden()

        dataloader = SessionParallelLoader(self.train_data, batch_size=self.args.batch_size,
                                           n_sample=self.args.n_sample, sample_alpha=self.args.sample_alpha)
        for input, target, session_mask in dataloader:
            # item indices, size=batch_size
            input = input.to(self.device)
            target = target.to(self.device)

            if self.args.location_mode != LocationMode.NONE:
                item_loc = dataloader.dataset.get_item_location(input)
                session_loc = dataloader.dataset.get_session_location(dataloader.curr_sess)
                # Diagonal because we only want the distance to the input items
                distance = calculate_session_item_distance(session_loc, item_loc).diagonal()
                distance = torch.FloatTensor(distance)
                distance.to(self.device)

            self.optim.zero_grad()
            # detach means the graph of operations will not be computed
            hidden = reset_hidden(hidden, session_mask).detach()
            if self.args.latent_mode != LatentMode.NONE:
                city_input = dataloader.dataset.get_city_index(input)
                city_input = torch.tensor(city_input, device=self.device)
                distr_input = dataloader.dataset.get_district_index(input)
                distr_input = torch.tensor(distr_input, device=self.device)
                logit, hidden, output = self.model(input, distance, hidden, city_input, distr_input)
            else:
                if self.args.location_mode == LocationMode.DISTHOT or self.args.location_mode == LocationMode.DISTANCE:
                    logit, hidden, output = self.model(input, distance, hidden)
                elif self.args.location_mode == LocationMode.CITYHOT:
                    city_input = dataloader.dataset.get_city_index(input)
                    city_input = torch.tensor(city_input, device=self.device)
                    logit, hidden, output = self.model(city_input, hidden)
                elif self.args.location_mode == LocationMode.FULLCONCAT:
                    city_input = dataloader.dataset.get_city_index(input)
                    city_input = torch.tensor(city_input, device=self.device)
                    distr_input = dataloader.dataset.get_district_index(input)
                    distr_input = torch.tensor(distr_input, device=self.device)
                    logit, hidden, output = self.model(city_input, hidden, distr_input, distance)
                elif self.args.location_mode > 5:
                    logit, hidden, output = self.model(input, distance, hidden)
                elif self.args.location_mode == LocationMode.CONCAT:
                    logit, hidden, output = self.model(input, hidden, distance)
                else:
                    logit, hidden, output = self.model(input, hidden)
            # sample output
            # this will create the following matrix
            # t_i is target for batch = i
            # o is logit for other items
            # [ t1,  o,  o]
            # [ o,  t2, o]
            # [ o,  o, t3]
            logit_sampled = logit[:, target.view(-1)]
            # if self.use_location:
            #     logit_sampled = self.compute_dist_logits(dataloader.get_item_location_from_index(target),
            #                                              dataloader.get_current_sess_location(),
            #                                              logit_sampled)
            loss = self.loss_func(logit_sampled)
            losses.append(loss.item())
            loss.backward()
            self.optim.step()

        mean_losses = np.mean(losses)
        return mean_losses
