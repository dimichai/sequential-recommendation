""" 
Created at 12/06/2019

@author: dimitris.michailidis
"""

import datetime
import json
import os

from sklearn.model_selection import ParameterSampler
import numpy as np

from experiments.olx_clicks.args import Args
from experiments.parallel_trainer import ParallelTrainer
from helpers.dataset import Dataset
from helpers.enums import LocationMode, ParallelMode
from models.base_gru import SimpleGRU
from models.city_gru import CityGRU
from models.dist_gru import DistanceGRU
from models.lossfunction import LossFunction
from models.optimizer import Optimizer
from models.parallel_model import ParallelModel

LOG_FILE = './logs/' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '_parallel' + '.txt'


class RandomSearch:
    def __init__(self, model: any, param_dist: dict, iterations: int):
        self.model = model
        self.param_dist = param_dist
        self.iterations = iterations

    # def run(self):
    # random_search = RandomizedSearchCV(self.model, self.param_dist, self.iterations)
    # sampler = ParameterSampler()


if __name__ == '__main__':
    # random_search = RandomSearch(None, )
    n_iter = 2
    n_epochs = 3
    logs = []

    # batch_size = [32, 64]
    batch_size = [16]
    hidden_size = [50, 100, 500, 1000]
    optimizer = ['Adagrad', 'Adam']
    loss_type = ['TOP1-max', 'BPR-max']
    parallel_mode = [ParallelMode.ENCODER, ParallelMode.DECODER, ParallelMode.HIDDEN]
    location_mode = [LocationMode.DISTHOT]
    learning_rate = np.random.uniform(0.001, 0.2, n_iter)

    param_dist = dict(batch_size=batch_size, lr=learning_rate, hidden_size=hidden_size, optimizer=optimizer,
                      loss_type=loss_type, parallel_mode=parallel_mode, location_mode=location_mode)

    current_iter = 0
    for param in list(ParameterSampler(param_dist, n_iter)):
        current_iter += 1
        print('Current Iteration is: ', current_iter)
        print(param)
        args = Args()
        args.batch_size = param['batch_size']
        args.hidden_size = param['hidden_size']
        args.optimizer_type = param['optimizer']
        args.loss_type = param['loss_type']
        args.lr = param['lr']
        args.n_epochs = n_epochs
        args.parallel_mode = param['parallel_mode']
        args.location_mode = param['location_mode']

        train_data = Dataset(os.path.join(args.data_folder, args.train_path))
        # valid_data = Dataset(os.path.join(DATA_FOLDER, VALID_PATH), itemmap=train_data.itemmap)
        valid_data = Dataset(os.path.join(args.data_folder, args.valid_path), itemmap=train_data.itemmap)

        # initialize hyper-parameters
        input_size = len(train_data.items)  # nr of unique items in the dataset
        output_size = input_size

        if args.location_mode == LocationMode.DISTHOT:
            location_model = DistanceGRU(input_size=input_size, hidden_size=args.hidden_size, output_size=output_size,
                                         num_layers=args.num_layers, dropout_gru=args.dropout_hidden,
                                         dropout_input=args.dropout_input, use_cuda=args.use_cuda,
                                         batch_size=args.batch_size, parallel_mode=args.parallel_mode)
        else:
            city_size = len(train_data.cities)
            distr_size = len(train_data.districts)
            location_model = CityGRU(city_size=city_size, distr_size=distr_size, hidden_size=args.hidden_size,
                                     output_size=output_size, num_layers=args.num_layers, dropout_gru=args.dropout_hidden,
                                     dropout_input=args.dropout_input, use_cuda=args.use_cuda,
                                     batch_size=args.batch_size,
                                     parallel_mode=args.parallel_mode, location_mode=args.location_mode)

        base_model = SimpleGRU(input_size=input_size, hidden_size=args.hidden_size, output_size=output_size,
                               num_layers=args.num_layers, dropout_gru=args.dropout_hidden, dropout_input=args.dropout_input,
                               use_cuda=args.use_cuda, batch_size=args.batch_size, embedding_dim=args.embedding_dim,
                               parallel_mode=args.parallel_mode, location_mode=args.location_mode)

        parallel_model = ParallelModel(input_size=args.hidden_size, output_size=base_model.output_size,
                                       use_cuda=args.use_cuda, parallel_mode=args.parallel_mode,
                                       combination_mode=args.combination_mode)

        base_optimizer = Optimizer(base_model.parameters(), optimizer_type=args.optimizer_type, lr=args.lr,
                                   weight_decay=args.weight_decay,
                                   momentum=args.momentum, eps=args.eps)
        location_optimizer = Optimizer(location_model.parameters(), optimizer_type=args.optimizer_type, lr=args.lr,
                                       weight_decay=args.weight_decay,
                                       momentum=args.momentum, eps=args.eps)

        parallel_optimizer = Optimizer(parallel_model.parameters(), optimizer_type=args.optimizer_type, lr=args.lr,
                                       weight_decay=args.weight_decay, momentum=args.momentum, eps=args.eps)

        loss_func = LossFunction(loss_type=args.loss_type, use_cuda=args.use_cuda)
        parallel_trainer = ParallelTrainer(base_model, location_model, parallel_model, train_data=train_data,
                                           eval_data=valid_data,
                                           base_optim=base_optimizer, dist_optim=location_optimizer,
                                           parallel_optim=parallel_optimizer,
                                           use_cuda=args.use_cuda, loss_func=loss_func, args=args)

        train_loss, loss, recall, mrr = parallel_trainer.train(start_epoch=0, end_epoch=n_epochs - 1, random_search=True)

        log_entry = {'location_mode': args.location_mode, 'parallel_mode': args.parallel_mode,
                     'combination_mode': args.combination_mode, 'batch_size': args.batch_size, 'lr': args.lr,
                     'hidden_size': args.hidden_size, 'optimizer': args.optimizer_type, 'loss_type': args.loss_type,
                     'train_loss': train_loss, 'valid_loss': loss, 'recall': recall, 'mrr': mrr}

        logs.append(log_entry)

    with open(LOG_FILE, 'a+') as file:
        json.dump(logs, file)
        file.write('\n')
