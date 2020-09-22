""" 
Created at 4/29/2019

@author: dimitris.michailidis
"""
import pandas as pd
import numpy as np

from helpers.distance_calculator import vectorized_haversine


class LocationPredictor:
    pop_list: pd.DataFrame
    item_coords: pd.DataFrame
    session_coords: pd.DataFrame

    def __init__(self, session_key='search_id', item_key='itemId', item_lat_key='itemLat', item_lon_key='itemLon',
                 user_lat_key='userLat', user_lon_key='userLong', top_n=100):
        self.session_key = session_key
        self.item_key = item_key
        self.item_lat_key = item_lat_key
        self.item_lon_key = item_lon_key
        self.user_lat_key = user_lat_key
        self.user_lon_key = user_lon_key
        self.top_n = top_n

    def prepare(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        """
        Trains the predictor.
        :return:
        """
        self.calc_pop_list(df_train)
        self.get_item_coords(df_train)
        self.get_session_coords(df_test)

    def calc_pop_list(self, data):
        """
        Calculates the list of top_n most popular items by item support.
        :param data:
        :return:
        """
        grp = data.groupby(self.item_key)
        self.pop_list = grp.size()
        self.pop_list = self.pop_list / (self.pop_list + 1)
        self.pop_list.sort_values(ascending=False, inplace=True)
        # self.pop_list = self.pop_list.head(self.top_n)

    def get_item_coords(self, data):
        """
        Get a list of all items and their coordinates, index is item_key
        :param data:
        :return:
        """
        grp = data.groupby(self.item_key)
        self.item_coords = grp.first()[[self.item_lat_key, self.item_lon_key]]

    def get_session_coords(self, data):
        """
        Get a list of all items and their coordinates, index is item_key
        :param data:
        :return:
        """
        grp = data.groupby(self.session_key)
        self.session_coords = grp.first()[[self.user_lat_key, self.user_lon_key]]
        pass

    def predict_next(self, session_id, input_item_id, predict_for_item_ids):
        """
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.

        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores.
            Every ID must be in the set of item IDs of the training set.

        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.

        """
        preds = np.zeros(len(predict_for_item_ids))
        session_coords = self.session_coords.loc[session_id]
        item_coords = self.item_coords.loc[predict_for_item_ids]
        dist = vectorized_haversine(session_coords[self.user_lat_key], session_coords[self.user_lon_key],
                                         item_coords[self.item_lat_key], item_coords[self.item_lon_key])
        dist = (1 / dist) / np.sum(1 / dist)  # normalize
        dist = dist.sort_values(ascending=False).head(self.top_n)

        # Get score based on the most popular items.
        # mask = np.in1d(predict_for_item_ids, self.pop_list.index)
        # preds[mask] = self.pop_list[predict_for_item_ids[mask]]

        # Enhance the score of the nearest items by the score of the distance.
        # mask = np.isin(predict_for_item_ids, dist.index)
        # preds[mask] += dist[predict_for_item_ids[mask]]

        # Predict the most popular TOP_N Nearest item
        mask = np.isin(predict_for_item_ids, dist.index)
        preds[mask] += self.pop_list[predict_for_item_ids[mask]]

        return pd.Series(data=preds, index=predict_for_item_ids)
