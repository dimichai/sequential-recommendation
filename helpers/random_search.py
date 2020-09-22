""" 
Created at 19/04/2019

@author: dimitris.michailidis
To perform random hyperparmeter search.
"""
import datetime
import json
import os

from sklearn.model_selection import ParameterSampler
import numpy as np

from experiments.olx_clicks.args import Args
from experiments.trainer import Trainer
from helpers.dataset import Dataset
from models.base_gru import SimpleGRU
from models.lossfunction import LossFunction
from models.optimizer import Optimizer

LOG_FILE = './logs/' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '_baseGRU' + '.txt'


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
    n_iter = 5
    n_epochs = 3
    logs = []

    # batch_size = [32, 64]
    batch_size = [10]
    hidden_size = [50, 100, 500, 1000]
    optimizer = ['RMSProp', 'Adagrad', 'Adadelta', 'Adam', 'SGD']
    loss_type = ['TOP1-max', 'BPR-max']
    learning_rate = np.random.uniform(0.001, 0.1, n_iter)

    param_dist = dict(batch_size=batch_size, lr=learning_rate, hidden_size=hidden_size, optimizer=optimizer,
                      loss_type=loss_type)

    for param in list(ParameterSampler(param_dist, n_iter)):
        # print(param)
        args = Args()
        args.batch_size = param['batch_size']
        args.hidden_size = param['hidden_size']
        args.optimizer_type = param['optimizer']
        args.loss_type = param['loss_type']
        args.lr = param['lr']
        args.n_epochs = n_epochs

        # print("Loading train data from {}".format(os.path.join(args.data_folder, args.train_path)))
        # print("Loading validation data from {}".format(os.path.join(args.data_folder, args.valid_path)))
        # print("Loading test data from {}\n".format(os.path.join(DATA_FOLDER, TEST_PATH)))

        train_data = Dataset(os.path.join(args.data_folder, args.train_path))
        # valid_data = Dataset(os.path.join(DATA_FOLDER, VALID_PATH), itemmap=train_data.itemmap)
        valid_data = Dataset(os.path.join(args.data_folder, args.valid_path), itemmap=train_data.itemmap)

        # initialize hyper-parameters
        input_size = len(train_data.items)  # nr of unique items in the dataset
        output_size = input_size

        model = SimpleGRU(input_size=input_size, hidden_size=args.hidden_size, output_size=output_size,
                          num_layers=args.num_layers, dropout_gru=args.dropout_hidden, dropout_input=args.dropout_input,
                          use_cuda=args.use_cuda, batch_size=args.batch_size, embedding_dim=args.embedding_dim,
                          parallel_mode=args.parallel_mode, location_mode=args.location_mode)

        optimizer = Optimizer(model.parameters(), optimizer_type=args.optimizer_type, lr=args.lr,
                              weight_decay=args.weight_decay, momentum=args.momentum, eps=args.eps)

        loss_func = LossFunction(loss_type=args.loss_type, use_cuda=args.use_cuda)
        trainer = Trainer(model, train_data=train_data, eval_data=valid_data, optim=optimizer,
                          use_cuda=args.use_cuda, loss_func=loss_func, args=args)

        train_loss, loss, recall, mrr = trainer.train(start_epoch=0, end_epoch=n_epochs - 1, random_search=True)

        log_entry = {'batch_size': args.batch_size, 'lr': args.lr, 'hidden_size': args.hidden_size,
                     'optimizer': args.optimizer_type, 'loss_type': args.loss_type, 'train_loss': train_loss,
                     'valid_loss': loss, 'recall': recall, 'mrr': mrr}

        logs.append(log_entry)

    with open(LOG_FILE, 'a+') as file:
        json.dump(logs, file)
        file.write('\n')
