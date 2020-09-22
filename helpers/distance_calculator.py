""" 
Created at 5/1/2019

@author: dimitris.michailidis
"""
import numpy as np


def vectorized_haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the haversine distance of vector coordinates
    :param lat1: latitudes of vector 1
    :param lon1: longitutes of vector 1
    :param lat2: latitudes of vector 2
    :param lon2: longitutes of vector 2
    :return: distance in kilometers
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def calculate_session_item_distance(session_loc, item_loc):
    lat1 = item_loc['itemLat'].values
    lon1 = item_loc['itemLon'].values
    lat2 = session_loc['userLat'].values
    lon2 = session_loc['userLong'].values

    return np.log(vectorized_haversine(lat1, lon1, lat2[:, None], lon2[:, None]))
