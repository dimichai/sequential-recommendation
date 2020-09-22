""" 
Created at 17/05/2019

@author: dimitris.michailidis
"""

import time

import numpy as np
import torch

from experiments.olx_clicks.args import Args
from helpers.dataset import Dataset
from helpers.distance_calculator import calculate_session_item_distance
from helpers.enums import ParallelMode, LocationMode
from helpers.model_loader import ModelLoader
from helpers.session_parallel_loader import SessionParallelLoader
from models.lossfunction import LossFunction
from models.optimizer import Optimizer
from models.parallel_evaluation import ParallelEvaluation
from models.parallel_model import ParallelModel


def reset_hidden(hidden, session_mask):
    """
    Resets the hidden state when a session terminates.
    :param hidden: the hidden state
    :param session_mask: the mask of the terminated sessions
    """
    if len(session_mask) != 0:
        hidden[:, session_mask, :] = 0
    return hidden


class ParallelTrainer:
    start_time: float

    def __init__(self, base_model, location_model, parallel_model: ParallelModel,
                 train_data: Dataset, eval_data: Dataset,
                 base_optim: Optimizer, dist_optim: Optimizer, parallel_optim: Optimizer, use_cuda: bool,
                 loss_func: LossFunction, args: Args, model_loader: ModelLoader = None):
        self.base_model = base_model
        self.location_model = location_model
        self.parallel_model = parallel_model

        self.train_data = train_data
        self.eval_data = eval_data
        self.base_optim = base_optim
        self.location_optim = dist_optim
        self.parallel_optim = parallel_optim
        # self.parallel_optim = Adagrad(self.parallel_model.parameters(), lr=args.lr)
        self.loss_func = loss_func

        self.evaluation = ParallelEvaluation(self.base_model, self.location_model, self.parallel_model, self.loss_func,
                                             use_cuda, batch_size=args.batch_size, location_mode=args.location_mode,
                                             parallel_mode=args.parallel_mode, k=args.topk)
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.args = args
        self.model_loader = model_loader

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

            print(
                f'Epoch: {epoch}, train-loss: {train_loss:.05}, test-loss: {loss:.05}, recall@{self.evaluation.topk}: {recall:.05}, '
                f'mrr@{self.evaluation.topk}: {mrr:.05}, recall@10: {recall10:.05}, mrr@10: {mrr10:.05}, recall@5: {recall5:.05}, mrr@5: {mrr5:.05}, '
                f'time: {time.time() - st}')

            # print("Epoch: {}, train-loss:{:.5f}, test-loss: {:.5f}, recall@{:d}: {:.5f}, mrr@{:d}: {:.5f}, time: {}"
            #       .format(epoch, train_loss, loss, self.evaluation.topk, recall, self.evaluation.topk, mrr, time.time() - st))

            checkpoint = {
                'model': self.parallel_model,
                'args': self.args,
                'epoch': epoch,
                'optim': self.base_optim,
                'loss': loss,
                'recall': recall,
                'mrr': mrr
            }

            if self.model_loader is not None:
                self.model_loader.save_model_checkpoint(checkpoint)

        if random_search:
            return train_loss, loss, recall, mrr

    def train_epoch(self):
        # set the model in train mode
        self.base_model.train()
        self.location_model.train()

        losses = []

        base_hidden = self.base_model.init_hidden()
        location_hidden = self.location_model.init_hidden()

        dataloader = SessionParallelLoader(self.train_data, batch_size=self.args.batch_size,
                                           n_sample=self.args.n_sample, sample_alpha=self.args.sample_alpha)
        for input, target, session_mask in dataloader:
            # item indices, size=batch_size
            input = input.to(self.device)
            target = target.to(self.device)

            # Prepare the distance model input
            item_loc = dataloader.dataset.get_item_location(input)
            session_loc = dataloader.dataset.get_session_location(dataloader.curr_sess)
            # Diagonal because we only want the distance to the input items
            distance = calculate_session_item_distance(session_loc, item_loc).diagonal()
            distance = torch.FloatTensor(distance)
            distance.to(self.device)

            self.base_optim.zero_grad()
            self.location_optim.zero_grad()
            self.parallel_optim.zero_grad()
            # detach means the graph of operations will not be computed
            base_hidden = reset_hidden(base_hidden, session_mask).detach()
            location_hidden = reset_hidden(location_hidden, session_mask).detach()

            base_logit, base_hidden, base_output = self.base_model(input, base_hidden)

            if self.args.location_mode == LocationMode.DISTHOT:
                location_logit, location_hidden, location_output = self.location_model(input, distance, location_hidden)
            elif self.args.location_mode == LocationMode.CITYHOT:
                city_input = dataloader.dataset.get_city_index(input)
                city_input = torch.tensor(city_input, device=self.device)
                location_logit, location_hidden, location_output = self.location_model(city_input, location_hidden)
            elif self.args.location_mode == LocationMode.DISTRICTHOT:
                distr_input = dataloader.dataset.get_district_index(input)
                distr_input = torch.tensor(distr_input, device=self.device)
                location_logit, location_hidden, location_output = self.location_model(distr_input, location_hidden)
            elif self.args.location_mode == LocationMode.FULLCONCAT:
                city_input = dataloader.dataset.get_city_index(input)
                city_input = torch.tensor(city_input, device=self.device)
                distr_input = dataloader.dataset.get_district_index(input)
                distr_input = torch.tensor(distr_input, device=self.device)
                location_logit, location_hidden, location_output = self.location_model(city_input, location_hidden, distr_input, distance)

            if self.args.parallel_mode == ParallelMode.ENCODER:
                merged_logit = self.parallel_model(base_output, location_output)
            elif self.args.parallel_mode == ParallelMode.DECODER:
                merged_logit = self.parallel_model(base_logit, location_logit)
            elif self.args.parallel_mode == ParallelMode.HIDDEN:
                merged_logit = self.parallel_model(base_hidden, location_hidden)

            logit_sampled = merged_logit[:, target.view(-1)]
            loss = self.loss_func(logit_sampled)
            losses.append(loss.item())
            loss.backward()

            self.parallel_optim.step()
            self.base_optim.step()
            self.location_optim.step()

        mean_losses = np.mean(losses)
        return mean_losses