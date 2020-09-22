""" 
Created at 10/04/2019

@author: dimitris.michailidis
based on https://github.com/hungthanhpham94/GRU4REC-pytorch
"""
import os
import string

import pandas as pd
import numpy as np


# use_loc = True


class Dataset(object):
    __item_idx_key = 'item_idx'
    __city_idx_key = 'city_idx'
    __sess_idx_key = 'sess_idx'
    __distr_idx_key = 'distr_idx'

    df: pd.DataFrame
    itemmap: pd.DataFrame
    sessmap: pd.DataFrame
    citymap: pd.DataFrame
    districtmap: pd.DataFrame
    __session_key: string
    __item_key: string
    __time_key: string
    session_index: np.ndarray
    session_offsets: np.ndarray

    def __init__(self, path, session_key='search_id', item_key='itemId', time_key='timestamp',
                 item_lat_key='itemLat', item_lon_key='itemLon', user_lat_key='userLat', user_lon_key='userLong',
                 item_city_key='itemCity', item_district_key='itemDistrict', itemmap=None, citymap=None,
                 districtmap=None):
        """
        :param path: string
            path to the data to load
        :param session_key: string
            key used to index the session id in the dataset
        :param item_key: string
            key used to index the item id in the dataset
        :param time_key: string
            key used to index the time key in the dataset
        """
        # Read the dataframe
        path_root, ext = os.path.splitext(path)
        if ext == '.csv':
            self.df = pd.read_csv(path)
        elif ext == '.pkl':
            self.df = pd.read_pickle(path)

        # self.df['itemDistrict'] = self.df['itemDistrict'].astype(int)

        # if use_loc:
        #     attr = pd.read_pickle('data/olx_train/clicks/item_freq_10/attributes.pkl')
        #     self.df = pd.merge(attr, self.df, on='itemId', how='inner')

        # Filter out not needed parameters
        # Todo: explore the possibility of removing 'position' from here and add variable instead
        self.df = self.df[[session_key, item_key, time_key, 'position',
                           item_lat_key, item_lon_key, item_city_key, item_district_key, user_lat_key, user_lon_key]]
        self.__session_key = session_key
        self.__item_key = item_key
        self.__time_key = time_key
        self.__item_lat_key = item_lat_key
        self.__item_lon_key = item_lon_key
        self.__item_city_key = item_city_key
        self.__item_district_key = item_district_key
        self.__user_lat_key = user_lat_key
        self.__user_lon_key = user_lon_key

        self.n_items = len(self.df[item_key].unique())

        self.create_item_city_map(citymap=citymap)
        self.create_item_district_map(districtmap=districtmap)
        self.create_item_map(itemmap=itemmap)
        self.create_session_map()
        # Sort sessions by the key, and then timestamp
        # Clicks within a session are next to each other and position-ordered.
        # self.df.sort_values([session_key], inplace=True)
        # self.df.sort_values([session_key, 'position'], inplace=True)
        # todo: change back
        self.df.sort_values([session_key], inplace=True)
        self.session_offsets = self.get_sessions_offset()
        self.session_index = self.get_ordered_session_index()

    def create_item_map(self, itemmap=None):
        """
        Creates an index for the unique items in the dataset. Then applies this index as a column on the dataframe.
        :param itemmap: string
            if there is an itemmap already available, pass it as a parameter and don't recompute it.
        """
        if itemmap is None:
            # Get the ids of the items.
            item_attr = self.df.groupby(self.__item_key).first()[
                [self.__item_lat_key, self.__item_lon_key, self.__city_idx_key, self.__distr_idx_key]].reset_index()
            # item_ids = self.df[self.__item_key].unique()
            item_ids = item_attr[self.__item_key].unique()
            # Create a series that gives item ids to an index e.g. index: 123124124, data: 0
            item_index = np.arange(len(item_ids))
            # Dataframe that maps item id to index [itemId, item_idx]
            itemmap = pd.DataFrame({
                self.__item_key: item_ids,
                self.__item_idx_key: item_index,
                self.__item_lat_key: item_attr[self.__item_lat_key],
                self.__item_lon_key: item_attr[self.__item_lon_key],
                self.__city_idx_key: item_attr[self.__city_idx_key],
                self.__distr_idx_key: item_attr[self.__distr_idx_key]})

        self.itemmap = itemmap

        # Merge the idx mapper to the original dataset
        self.df = pd.merge(self.df, self.itemmap[[self.__item_key, self.__item_idx_key]],
                           on=self.__item_key, how='inner')

    def create_session_map(self):
        # map session id to attributes
        sess_data = self.df.groupby(self.__session_key).first()[
            [self.__user_lat_key, self.__user_lon_key]].reset_index()
        sess_ids = sess_data[self.__session_key].unique()
        sess_index = np.arange(len(sess_ids))
        sessmap = pd.DataFrame({
            self.__session_key: sess_ids,
            self.__sess_idx_key: sess_index,
            self.__user_lat_key: sess_data[self.__user_lat_key],
            self.__user_lon_key: sess_data[self.__user_lon_key]})
        self.sessmap = sessmap

        # Merge the idx mapper to the original dataset
        self.df = pd.merge(self.df, self.sessmap[[self.__session_key, self.__sess_idx_key]],
                           on=self.__session_key, how='inner')

    def create_item_city_map(self, citymap=None):
        if citymap is None:
            # Get the ids of the cities.
            city_data = self.df.groupby(self.__item_city_key).first()[
                [self.__item_lat_key, self.__item_lon_key]].reset_index()
            city_ids = city_data[self.__item_city_key].unique()
            city_index = np.arange(len(city_ids))
            citymap = pd.DataFrame({
                self.__item_city_key: city_ids,
                self.__city_idx_key: city_index,
            })

        self.citymap = citymap
        self.df = pd.merge(self.df, self.citymap[[self.__item_city_key, self.__city_idx_key]],
                           on=self.__item_city_key, how='inner')

    def create_item_district_map(self, districtmap=None):
        if districtmap is None:
            # Get the ids of the districts.
            distr_data = self.df.groupby(self.__item_district_key).first()[
                [self.__item_lat_key, self.__item_lon_key]].reset_index()
            distr_ids = distr_data[self.__item_district_key].unique()
            distr_index = np.arange(len(distr_ids))
            districtmap = pd.DataFrame({
                self.__item_district_key: distr_ids,
                self.__distr_idx_key: distr_index,
            })

        self.districtmap = districtmap
        self.df = pd.merge(self.df, self.districtmap[[self.__item_district_key, self.__distr_idx_key]],
                           on=self.__item_district_key, how='inner')

    def get_sessions_offset(self) -> np.ndarray:
        """
        Get the index offset of the sessions. Essentially where does each session's actions start.
        """
        offsets = np.zeros(self.df[self.__session_key].nunique() + 1, dtype=np.int32)
        offsets[1:] = self.df.groupby(self.__session_key).size().cumsum()
        return offsets

    def get_ordered_session_index(self) -> np.ndarray:
        """
        Returns the index of the sessions, sorted by the timestamp they were initialized.
        :return:
        """
        # Get the starting time of each session
        # TODO: reconsider the whole position thing.
        session_start_time = self.df.groupby(self.__session_key)['timestamp'].min().values
        # Get the indices that would sort the array by the start time.
        sorted_index = np.argsort(session_start_time)
        return sorted_index

    @property
    def items(self) -> pd.DataFrame:
        """
        Get the ids of the unique items in the dataset
        :return: pd.DataFrame
            the ids of the items in the dataset
        """
        return self.itemmap.itemId.unique()

    @property
    def cities(self) -> pd.DataFrame:
        """
        :return: pd.DataFrame
            ids of the cities in the dataset
        """
        return self.citymap[self.__item_city_key].unique()

    @property
    def districts(self) -> pd.DataFrame:
        """
        :return: pd.DataFrame
            ids of the districts in the dataset
        """
        return self.districtmap[self.__item_district_key].unique()

    def get_session_location(self, session_index):
        return self.sessmap.iloc[session_index].reset_index()[[self.__user_lat_key, self.__user_lon_key]]

    def get_item_location(self, item_index=None):
        if item_index is not None:
            return self.itemmap.iloc[item_index.cpu()].reset_index()[[self.__item_lat_key, self.__item_lon_key]]
        else:
            return self.itemmap[[self.__item_lat_key, self.__item_lon_key]]

    def get_city_index(self, item_index):
        return self.itemmap.iloc[item_index.cpu()][self.__city_idx_key].tolist()

    def get_district_index(self, item_index):
        return self.itemmap.iloc[item_index.cpu()][self.__distr_idx_key].tolist()
