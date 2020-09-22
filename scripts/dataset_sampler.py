""" 
Created at 5/1/2019

@author: dimitris.michailidis
To create small samples from the train/test dataset to use for debugging/testing, etc.
"""

import pandas as pd
import numpy as np

TRAIN_INPUT = './data/olx_train/clicks/item_freq_10/train10.pkl'
TRAIN_OUTPUT = './data/olx_train/clicks/samples2/train10.csv'

TEST_INPUT = './data/olx_train/clicks/item_freq_10/test10.pkl'
TEST_OUTPUT = './data/olx_train/clicks/samples2/test10.csv'

ATTR_FILE = './data/olx_train/clicks/item_freq_10/attributes.pkl'

# number of random sessions to sample
n_train_sessions = 100
n_test_sessions = 20


def sample_sessions(df, size):
    samples = np.random.choice(df.search_id.unique(), size=size)
    return df[df.search_id.isin(samples)]


if __name__ == '__main__':
    np.random.seed(7)

    # Load the dataframes
    df_train = pd.read_pickle(TRAIN_INPUT)
    df_test = pd.read_pickle(TEST_INPUT)
    df_attributes = pd.read_pickle(ATTR_FILE)

    # Merge dataframes with the item attributes
    df_train = pd.merge(df_attributes, df_train, on='itemId', how='inner')
    df_test = pd.merge(df_attributes, df_test, on='itemId', how='inner')

    # Sample sessions from the train set
    df_train_samples = sample_sessions(df_train, n_train_sessions)

    # Filter out sessions from the test set that do not contain items clicked in the train set
    items = df_train_samples.itemId.unique()
    df_test = df_test[df_test.itemId.isin(items)]

    # Filter out sessions of length < 2 from the test set
    tslength = df_test.groupby('search_id').size()
    df_test = df_test[np.isin(df_test.search_id, tslength[tslength >= 2].index)]

    # Sample the test set
    df_test_samples = sample_sessions(df_test, 20)

    # Save the dataframes to csv files
    df_train_samples.to_csv(TRAIN_OUTPUT, index=False)
    df_test_samples.to_csv(TEST_OUTPUT, index=False)
