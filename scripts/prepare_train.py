"""
Created at 28/06/2019

@author: dimitris.michailidis
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import datetime
import os

INPUT_PATH = '../data/olx/clicks/clean/'
TRAIN_PATH = '../data/olx_train/clicks/clean/train.pkl'
TEST_PATH = '../data/olx_train/clicks/clean/test.pkl'

spark = SparkSession.builder \
    .master('local') \
    .appName('Click Prediction') \
    .config('spark.executor.memory', '5g') \
    .config('spark.driver.memory', '5g') \
    .getOrCreate()

df = spark.read.parquet(INPUT_PATH)
df = df.cache()

# Keep the last 24 hours of impressions as a test set - use the rest for train
test_threshold = df.agg(F.max('timestamp')).head()['max(timestamp)'] - datetime.timedelta(1)
train = df.where(df.timestamp < test_threshold)
test = df.where(df.timestamp >= test_threshold)
# filter out actions involving items that are not present in the training set
test = test.join(train, ['itemId'], how='leftsemi')

print('Full train set\n\tActions: {}\n\tSessions: {}\n\tItems: {}' \
      .format(train.count(), train.agg(F.countDistinct('search_id')).collect(), train.agg(F.countDistinct('itemId')).collect()))

print('Test set\n\tActions: {}\n\tSessions: {}\n\tItems: {}'
      .format(test.count(), test.agg(F.countDistinct('search_id')).collect(), test.agg(F.countDistinct('itemId')).collect()))

directory = os.path.dirname(TRAIN_PATH)
if not os.path.exists(directory):
    os.makedirs(directory)

train.toPandas().to_pickle(TRAIN_PATH)
test.toPandas().to_pickle(TEST_PATH)
