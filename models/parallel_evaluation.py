""" 
Created at 20/05/2019

@author: dimitris.michailidis
"""

import numpy as np
import torch

from helpers.dataset import Dataset
from helpers.distance_calculator import calculate_session_item_distance
from helpers.enums import LocationMode, ParallelMode
from helpers.session_parallel_loader import SessionParallelLoader
from models.lossfunction import LossFunction
from models.metrics import evaluate
from models.parallel_model import ParallelModel


class ParallelEvaluation:
    def __init__(self, base_model, dist_model, parallel_model: ParallelModel, loss_func: LossFunction, use_cuda: bool,
                 k=20, batch_size=50, location_mode=LocationMode.NONE, parallel_mode=ParallelMode.NONE):
        self.base_model = base_model
        self.location_model = dist_model
        self.paralell_model = parallel_model
        self.parallel_mode = parallel_mode

        self.loss_func = loss_func
        self.topk = k
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.batch_size = batch_size
        self.location_mode = location_mode

    def eval(self, eval_data: Dataset):
        # set the model to evaluation mode (effects on dropout, batch normalization, etc.)
        self.base_model.eval()
        self.location_model.eval()
        self.paralell_model.eval()

        losses = []
        recalls = []
        mrrs = []
        recalls5 = []
        mrrs5 = []
        recalls10 = []
        mrrs10 = []

        dataloader = SessionParallelLoader(eval_data, batch_size=self.batch_size)
        # evaluation mode - no need to compute and store gradients
        with torch.no_grad():
            base_hidden = self.base_model.init_hidden()
            location_hidden = self.location_model.init_hidden()
            for input, target, mask in dataloader:
                input = input.to(self.device)
                target = target.to(self.device)

                item_loc = dataloader.dataset.get_item_location(input)
                session_loc = dataloader.dataset.get_session_location(dataloader.curr_sess)
                # Diagonal because we only want the distance to the input items
                distance = calculate_session_item_distance(session_loc, item_loc).diagonal()
                distance = torch.FloatTensor(distance)
                distance.to(self.device)

                base_logit, base_hidden, base_output = self.base_model(input, base_hidden)

                if self.location_mode == LocationMode.DISTHOT or self.location_mode == LocationMode.DISTANCE:
                    location_logit, location_hidden, location_output = self.location_model(input, distance, location_hidden)
                elif self.location_mode == LocationMode.CITYHOT:
                    city_input = dataloader.dataset.get_city_index(input)
                    city_input = torch.tensor(city_input, device=self.device)
                    location_logit, location_hidden, location_output = self.location_model(city_input, location_hidden)
                elif self.location_mode == LocationMode.DISTRICTHOT:
                    distr_input = dataloader.dataset.get_district_index(input)
                    distr_input = torch.tensor(distr_input, device=self.device)
                    location_logit, location_hidden, location_output = self.location_model(distr_input, location_hidden)
                elif self.location_mode == LocationMode.FULLCONCAT:
                    city_input = dataloader.dataset.get_city_index(input)
                    city_input = torch.tensor(city_input, device=self.device)
                    distr_input = dataloader.dataset.get_district_index(input)
                    distr_input = torch.tensor(distr_input, device=self.device)
                    location_logit, location_hidden, location_output = self.location_model(city_input, location_hidden,
                                                                                           distr_input, distance)

                if self.parallel_mode == ParallelMode.ENCODER:
                    merged_logit = self.paralell_model(base_output, location_output)
                elif self.parallel_mode == ParallelMode.DECODER:
                    merged_logit = self.paralell_model(base_logit, location_logit)
                elif self.parallel_mode == ParallelMode.HIDDEN:
                    merged_logit = self.paralell_model(base_hidden, location_hidden)

                logit_sampled = merged_logit[:, target.view(-1)]
                loss = self.loss_func(logit_sampled)
                # replace new logit on the total one to calculate the metrics
                # this will have an effect only if we change logit_sample by using location.
                # logit[:, target.view(-1)] = logit_sampled
                recall, mrr, hits = evaluate(merged_logit, target, k=self.topk)
                recall5, mrr5, hits = evaluate(merged_logit, target, k=5)
                recall10, mrr10, hits = evaluate(merged_logit, target, k=10)

                # torch.Tensor.item() to get a Python number from a tensor containing a single value
                losses.append(loss.item())
                recalls.append(recall)
                mrrs.append(mrr.item())

                recalls5.append(recall5)
                mrrs5.append(mrr5.item())

                recalls10.append(recall10)
                mrrs10.append(mrr10.item())

        mean_losses = np.mean(losses)
        mean_recall = np.mean(recalls)
        mean_mrr = np.mean(mrrs)

        mean_recall5 = np.mean(recalls5)
        mean_mrr5 = np.mean(mrrs5)

        mean_recall10 = np.mean(recalls10)
        mean_mrr10 = np.mean(mrrs10)

        return mean_losses, mean_recall, mean_mrr, mean_recall5, mean_mrr5, mean_recall10, mean_mrr10


