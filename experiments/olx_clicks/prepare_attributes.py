""" 
Created at 4/29/2019

@author: dimitris.michailidis
Copy the needed .json files for the training and test sets.
Json files contain attributes about the listings and are in the format itemId.json
"""
import urllib
from shutil import copyfile
import os
import pandas as pd
import json
import numpy as np

DATA_DIR = './data/olx_train/clicks/clean/'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.pkl')
TEST_PATH = os.path.join(DATA_DIR, 'test.pkl')
JSON_INPUT = './data/olx/listings/json/'
JSON_OUTPUT = os.path.join(DATA_DIR, 'json')
NOTFOUND_FILE = os.path.join(DATA_DIR, 'json/notfound.txt')


def copy_json_files(df: pd.DataFrame):
    """
    Json files containing the item attributes are downloaded from S3; each item has its own .json file.
    This copies the relevant json files from the raw download directory into the training directory.
    It also does create a list of non-found files that then saves in a file for further use.
    :param df: the dataframe for which we want to copy relevant files.
    """
    total_items = df['itemId'].nunique()
    current_item = 0
    notfound_items = []

    for item_id in df['itemId'].unique():
        current_item += 1
        print('Copying file ', current_item, ' of ', total_items)

        item_id = str(item_id)
        try:
            copyfile(os.path.join(JSON_INPUT, item_id + '.json'), os.path.join(JSON_OUTPUT, item_id + '.json'))
        except FileNotFoundError:
            notfound_items.append(item_id)

    print('Not found items count: ', len(notfound_items))
    print('Not found items list: ', notfound_items)

    with open(NOTFOUND_FILE, 'w+') as f:
        for item in notfound_items:
            f.write("%s\n" % item)


def extract_attributes_from_json(df: pd.DataFrame):
    """
    Reads the copied json files for each item and extracts relevant attributes (long, lat, text, etc.)
    Then creates a dataframe out of these and saves them in a pickle file.
    :param df:
    :return:
    """
    # File that contains the ids of the items for which an attribute json file is not found.
    notfound_file = open(NOTFOUND_FILE, 'r')
    notfound = [line.rstrip('\n') for line in notfound_file.readlines()]

    df_known_items = df[~df['itemId'].isin(notfound)]
    total_items = df_known_items['itemId'].nunique()
    current_item = 0
    attributes = []

    for item_id in df_known_items['itemId'].unique():
        current_item += 1
        print('Reading attributes from ', current_item, ' of ', total_items)

        item_id = str(item_id)
        with open(os.path.join(JSON_OUTPUT, item_id + '.json')) as json_file:
            data_json = json.load(json_file)
            dict_attr = {'itemId': data_json['data']['id'],
                         'itemTitle': data_json['data']['title'],
                         'itemCategory': data_json['data']['category_id'],
                         # 'itemDesc': data_json['data']['description'],
                         'itemLon': data_json['data']['locations'][0]['lon'],
                         'itemLat': data_json['data']['locations'][0]['lat'],
                         'itemCity': data_json['data']['locations'][0]['city_id'],
                         'itemDistrict': data_json['data']['locations'][0]['district_id']}

            attributes.append(dict_attr)

    df_attr = pd.DataFrame(attributes)
    # df_attr.loc[df_attr.itemDistrict == '', 'itemDistrict'] = df_attr[df_attr.itemDistrict == '']['itemCity']
    df_attr.loc[df_attr.itemDistrict == '', 'itemDistrict'] = df_attr['itemCity']

    df_attr = df_attr.astype({'itemId': np.int64, 'itemLat': np.float, 'itemLon': np.float, 'itemDistrict': np.int64,
                              'itemCity': np.int64})
    df_attr.to_pickle(os.path.join(DATA_DIR, 'attributes.pkl'))


def download_missing_json_files():
    """
    Goes through the list of the non-found files in the repository and downloads them from the olx api.
    """
    olx_endpoint = 'https://api.olx.co.za/api/v1/items/'
    notfound_file = open(NOTFOUND_FILE, 'r')
    notfound = [line.rstrip('\n') for line in notfound_file.readlines()]
    total_items = len(notfound)
    current_item = 0

    for item_id in notfound:
        current_item += 1
        print('Downloading item ', current_item, ' of ', total_items)
        item_id = item_id.rstrip('\n')
        url = olx_endpoint + item_id
        filepath = os.path.join(JSON_INPUT, item_id + '.json')
        filename, fileinfo = urllib.request.urlretrieve(url, filepath)


def attach_attributes_to_dataset(df_dataset: pd.DataFrame, df_attributes: pd.DataFrame):
    """
    Merges the item attributes dataframe into the main dataset.
    :param df_dataset: the main dataset
    :param df_attributes: the dataframe containing the item attributes
    """
    return pd.merge(df_attributes, df_dataset, on='itemId', how='inner')


if __name__ == '__main__':
    df_train = pd.read_pickle(TRAIN_PATH)

    copy_json_files(df_train)
    download_missing_json_files()

    extract_attributes_from_json(df_train)
    # Merge attributes to train/test set
    train = pd.read_pickle(TRAIN_PATH)
    test = pd.read_pickle(TEST_PATH)
    attributes = pd.read_pickle((os.path.join(DATA_DIR, 'attributes.pkl')))

    train = attach_attributes_to_dataset(train, attributes)
    test = attach_attributes_to_dataset(test, attributes)

    train.to_pickle(os.path.join(DATA_DIR, 'train10_attr.pkl'))
    test.to_pickle(os.path.join(DATA_DIR, 'test10_attr.pkl'))
    # print('Done!')
