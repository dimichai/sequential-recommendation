"""
Created at 12/04/2019

@author: dimitris.michailidis
original: https://github.com/hungthanhpham94/GRU4REC-pytorch
"""
import csv
import datetime

import numpy as np
import torch

from helpers.dataset import Dataset
from helpers.distance_calculator import calculate_session_item_distance
from helpers.enums import LocationMode, LatentMode
from helpers.session_parallel_loader import SessionParallelLoader
from models.lossfunction import LossFunction
from models.metrics import evaluate

LOG_FILE = './logs/' + datetime.datetime.now().strftime("%m%d_%H%M%S") + '_train' + '.csv'
SESS_LOG_FILE = './logs/' + datetime.datetime.now().strftime("%m%d_%H%M%S") + '_sessions' + '.csv'


class Evaluation:
    def __init__(self, model, loss_func: LossFunction, use_cuda: bool, k=20, batch_size=50,
                 location_mode=LocationMode.NONE, latent_mode=LatentMode.NONE):
        self.model = model
        self.loss_func = loss_func
        self.topk = k
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.batch_size = batch_size
        self.location_mode = location_mode
        self.latent_mode = latent_mode

    def eval(self, eval_data: Dataset):
        # set the model to evaluation mode (effects on dropout, batch normalization, etc.)
        self.model.eval()

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
            hidden = self.model.init_hidden()
            for input, target, mask in dataloader:
                input = input.to(self.device)
                target = target.to(self.device)

                if self.location_mode != LocationMode.NONE:
                    item_loc = dataloader.dataset.get_item_location(input)
                    session_loc = dataloader.dataset.get_session_location(dataloader.curr_sess)
                    # Diagonal because we only want the distance to the input items
                    distance = calculate_session_item_distance(session_loc, item_loc).diagonal()
                    distance = torch.FloatTensor(distance)
                    distance.to(self.device)

                if self.latent_mode != LatentMode.NONE:
                    city_input = dataloader.dataset.get_city_index(input)
                    city_input = torch.tensor(city_input, device=self.device)
                    distr_input = dataloader.dataset.get_district_index(input)
                    distr_input = torch.tensor(distr_input, device=self.device)
                    logit, hidden, output = self.model(input, distance, hidden, city_input, distr_input)
                else:
                    if self.location_mode != LocationMode.NONE:
                        # item_loc = dataloader.dataset.get_item_location(input)
                        # session_loc = dataloader.dataset.get_session_location(dataloader.curr_sess)
                        # Diagonal because we only want the distance to the input items
                        # distance = calculate_session_item_distance(session_loc, item_loc).diagonal()
                        # distance = torch.FloatTensor(distance)
                        # distance.to(self.device)

                        if self.location_mode == LocationMode.DISTHOT:
                            logit, hidden, output = self.model(input, distance, hidden)
                        elif self.location_mode == LocationMode.CITYHOT:
                            city_input = dataloader.dataset.get_city_index(input)
                            distr_input = dataloader.dataset.get_district_index(input)
                            # city_input = torch.tensor(city_input)
                            city_input = torch.tensor(city_input, device=self.device)
                            logit, hidden, output = self.model(city_input, hidden)
                        elif self.location_mode == LocationMode.FULLCONCAT:
                            city_input = dataloader.dataset.get_city_index(input)
                            city_input = torch.tensor(city_input, device=self.device)

                            distr_input = dataloader.dataset.get_district_index(input)
                            distr_input = torch.tensor(distr_input, device=self.device)
                            logit, hidden, output = self.model(city_input, hidden, distr_input, distance)
                        elif self.location_mode > 5:
                            logit, hidden, output = self.model(input, distance, hidden)
                        elif self.location_mode == LocationMode.CONCAT:
                            logit, hidden, output = self.model(input, hidden, distance)
                    else:
                        logit, hidden, output = self.model(input, hidden)

                logit_sampled = logit[:, target.view(-1)]
                loss = self.loss_func(logit_sampled)
                # replace new logit on the total one to calculate the metrics
                # this will have an effect only if we change logit_sample by using location.
                # logit[:, target.view(-1)] = logit_sampled
                recall, mrr, hits = evaluate(logit, target, k=self.topk)
                recall5, mrr5, hits5 = evaluate(logit, target, k=5)
                recall10, mrr10, hits10 = evaluate(logit, target, k=10)

                ranks = hits[:, 1]
                hits = hits[:, 0]
                hits5 = hits5[:, 0]
                hits10 = hits10[:, 0]

                # torch.Tensor.item() to get a Python number from a tensor containing a single value
                losses.append(loss.item())
                recalls.append(recall)
                mrrs.append(mrr.item())

                recalls5.append(recall5)
                mrrs5.append(mrr5.item())

                recalls10.append(recall10)
                mrrs10.append(mrr10.item())

                with open(SESS_LOG_FILE, 'a+', newline='') as file:
                    writer = csv.writer(file)
                    for i in range(len(dataloader.curr_sess)):
                        if i in hits:
                            rank_index = (hits == i).nonzero().item()
                            writer.writerow([dataloader.curr_sess[i], ranks[rank_index].item()])
                            # writer.writerow([dataloader.curr_sess[i], 1])
                        else:
                            writer.writerow([dataloader.curr_sess[i], 0])

        with open(LOG_FILE, 'a+', newline='') as file:
            writer = csv.writer(file)
            for i in range(len(losses)):
                writer.writerow([losses[i], recalls[i], mrrs[i]])

        mean_losses = np.mean(losses)
        mean_recall = np.mean(recalls)
        mean_mrr = np.mean(mrrs)

        mean_recall5 = np.mean(recalls5)
        mean_mrr5 = np.mean(mrrs5)

        mean_recall10 = np.mean(recalls10)
        mean_mrr10 = np.mean(mrrs10)

        return mean_losses, mean_recall, mean_mrr, mean_recall5, mean_mrr5, mean_recall10, mean_mrr10
