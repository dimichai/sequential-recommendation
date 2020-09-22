""" 
Created at 09/04/2019

@author: balasz hidasi
@author: dimitris.michailidis
"""

import numpy as np


def evaluate_sessions(pr, test_data, train_data, items=None, cut_off=20, session_key='search_id', item_key='itemId',
                      time_key='timestamp'):
    """
    Evaluates the baselines wrt. recommendation accuracy measured by recall@N and MRR@N. Has no batch evaluation capabilities. Breaks up ties.

    Parameters
    --------
    pr : baseline predictor
        A trained instance of a baseline predictor.
    test_data : pandas.DataFrame
        Test data. It contains the transactions of the test set.It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    train_data : pandas.DataFrame
        Training data. Only required for selecting the set of item IDs of the training set.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to. If None, all items of the training set are used. Default value is None.
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')

    Returns
    --------
    out : tuple
        (Recall@N, MRR@N)

    """
    test_data.sort_values([session_key, time_key], inplace=True)
    items_to_predict = train_data[item_key].unique()
    evalutation_point_count = 0
    prev_iid, prev_sid = -1, -1
    mrr, recall, mrr5, recall5, mrr10, recall10 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(len(test_data)):
        sid = test_data[session_key].values[i]
        iid = test_data[item_key].values[i]
        if prev_sid != sid:
            prev_sid = sid
        else:
            if items is not None:
                if np.in1d(iid, items):
                    items_to_predict = items
                else:
                    items_to_predict = np.hstack(([iid], items))
            preds = pr.predict_next(sid, prev_iid, items_to_predict)
            preds[np.isnan(preds)] = 0
            preds += 1e-8 * np.random.rand(len(preds))  # Breaking up ties
            rank = (preds > preds[iid]).sum() + 1
            assert rank > 0
            if rank < cut_off:
                recall += 1
                mrr += 1.0 / rank
            if rank < 10:
                recall10 += 1
                mrr10 += 1.0 / rank
            if rank < 5:
                recall5 += 1
                mrr5 += 1.0 / rank

            evalutation_point_count += 1
            print('Evaluated: ', i, 'Recall@20: ', recall / evalutation_point_count, 'MRR@20: ',
                  mrr / evalutation_point_count,
                  'Recall@10: ', recall10 / evalutation_point_count, 'MRR@10: ', mrr10 / evalutation_point_count,
                  'Recall@5: ', recall5 / evalutation_point_count, 'MRR@5: ', mrr5 / evalutation_point_count)
        prev_iid = iid
    return recall / evalutation_point_count, mrr / evalutation_point_count, \
           recall10 / evalutation_point_count, mrr10 / evalutation_point_count, \
           recall5 / evalutation_point_count, mrr5 / evalutation_point_count
