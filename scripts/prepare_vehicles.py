"""
Created at 28/06/2019

@author: dimitris.michailidis
"""
from helpers.distance_calculator import vectorized_haversine
import pandas as pd

df_train = pd.read_pickle('../data/olx_train/clicks/clean/train.pkl')
df_test = pd.read_pickle('../data/olx_train/clicks/clean/test.pkl')

df_train['distance'] = vectorized_haversine(df_train.itemLat, df_train.itemLon, df_train.userLat, df_train.userLong)
df_test['distance'] = vectorized_haversine(df_test.itemLat, df_test.itemLon, df_test.userLat, df_test.userLong)

df_train = df_train[df_train.itemCategory.isin(['378', '379'])]
df_test = df_test[df_test.itemCategory.isin(['378', '379'])]

df_train.to_pickle('../data/olx_train/clicks/vehicles/vehicles_train.pkl')
df_test.to_pickle('../data/olx_train/clicks/vehicles/vehicles_test.pkl')
