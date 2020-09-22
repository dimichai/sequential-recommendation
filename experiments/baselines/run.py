""" 
Created at 09/04/2019

@author: dimitris.michailidis
To train and evaluate popular, non-rnn baselines
"""
from experiments.baselines.evaluation import evaluate_sessions
from experiments.baselines.item_knn import ItemKNN
from experiments.baselines.location_only import LocationPredictor
from experiments.baselines.pop import Pop
from experiments.baselines.session_pop import SessionPop
import pandas as pd
import numpy as np

import sys
sys.path.append('.')

debug = False

if not debug:
    TRAIN_PATH = '../../data/olx_train/clicks/vehicles/vehicles_train.pkl'
    TEST_PATH = '../../data/olx_train/clicks/vehicles/vehicles_test.pkl'
    ATTR_PATH = '../../data/olx_train/clicks/clean/attributes.pkl'
else:
    TRAIN_PATH = '../../data/olx_train/clicks/samples/train10.csv'
    TEST_PATH = '../../data/olx_train/clicks/samples/test10.csv'

if __name__ == '__main__':
    if not debug:
        train = pd.read_pickle(TRAIN_PATH)
        test = pd.read_pickle(TEST_PATH)
        # attr = pd.read_pickle(ATTR_PATH)

        # train = pd.merge(attr, train, on='itemId', how='inner')
        # test = pd.merge(attr, test, on='itemId', how='inner')
    else:
        train = pd.read_csv(TRAIN_PATH)
        test = pd.read_csv(TEST_PATH)

    train = train[['search_id', 'itemId', 'timestamp', 'position', 'userLat', 'userLong', 'itemLat', 'itemLon']]
    test = test[['search_id', 'itemId', 'timestamp', 'position', 'userLat', 'userLong', 'itemLat', 'itemLon']]
    # filter out sessions in the test set whose items are not in the training set.
    test = test[np.isin(test.itemId, train.itemId)]
    # sort by session sequence
    train.sort_values(['search_id', 'position'], inplace=True)

    # ------------------ POP ------------------
    # pop = Pop(top_n=20)
    # pop.fit(train)
    # print('POP Fitted.')
    # recall, mrr, recall10, mrr10, recall5, mrr5 = evaluate_sessions(pop, test, train)
    # print('POP recall@20 is ', recall)
    # print('POP mrr@20 is ', mrr)
    # print('POP recall@10 is ', recall10)
    # print('POP mrr@10 is ', mrr10)
    # print('POP recall@5 is ', recall5)
    # print('POP mrr@5 is ', mrr5)

    # ------------------ SPOP ------------------
    # spop = SessionPop(top_n=20)
    # print('Running the SPOP baseline.')
    # spop.fit(train)
    # print('SPOP fitted.')
    #
    # recall, mrr, recall10, mrr10, recall5, mrr5 = evaluate_sessions(spop, test, train)
    # print('SPOP recall@20 is ', recall)
    # print('SPOP mrr@20 is ', mrr)
    #
    # print('SPOP recall@10 is ', recall10)
    # print('SPOP mrr@10 is ', mrr10)
    #
    # print('SPOP recall@5 is ', recall5)
    # print('SPOP mrr% is ', mrr5)

    # ------------------ ItemKNN ------------------
    # knn = ItemKNN()
    # knn.fit(train)
    # print('KNN Fitted.')
    # recall, mrr, recall10, mrr10, recall5, mrr5 = evaluate_sessions(knn, test, train, cut_off=20)
    # print('KNN recall@20 is ', recall)
    # print('KNN mrr@20 is ', mrr)
    #
    # print('KNN recall@10 is ', recall10)
    # print('KNN mrr@10 is ', mrr10)
    #
    # print('KNN recall@5 is ', recall5)
    # print('KNN mrr@5 is ', mrr5)

    # ------------------ Location Based ------------------
    loc = LocationPredictor()
    if debug:
        loc = LocationPredictor(top_n=100)
    loc.prepare(train, test)
    print('LOC Prepared.')
    recall, mrr, recall10, mrr10, recall5, mrr5 = evaluate_sessions(loc, test, train, cut_off=20)
    print('LOC recall@20 is ', recall)
    print('LOC mrr@20 is ', mrr)

    print('LOC recall@10 is ', recall10)
    print('LOC mrr@10 is ', mrr10)

    print('LOC recall@5 is ', recall5)
    print('LOC mrr@5 is ', mrr5)
