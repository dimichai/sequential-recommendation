# Created by dimitris.michailidis at 03/04/2019
# To filter, manipulate and save subsets of the original OLX dataset.
# Reads from parquet, writes back to parquet
import string
from typing import Any

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def filter_by_value(df, column_name: string, value: Any):
    """
    Filters the given dataframe by a column value.
    :param df: spark DataFrame
        the dataframe to filter
    :param column_name: string
        the name of the column to filter on
    :param value: Any
        the value required for the items in the column to be kept
    :return: spark DataFrame
        the filtered DataFrame
    """
    return df.filter(df[column_name] == value)


def filter_by_item_support(df, item_key='itemId', support=10):
    """
    Filters the given dataframe by the frequency of appearance of the items.
    Keeps only items that appear at least 'support' times.
    :param df: spark DataFrame
        the DataFrame to filter
    :param item_key: string
        column name of the item data
    :param support: int
        minimum item frequency required for an item to be kept
    :return: the filtered dataset
    """
    return df.join(
        df.groupBy(item_key).count().filter(F.col('count') >= support).select(item_key),
        on=item_key, how='inner')


def filter_by_session_length(df, session_key='search_id', minlen=2):
    """
    Filters the given dataframe by the session length. Keeps only sessions with at least 'minlen' actions.
    :param df: spark DataFrame
        the DataFrame to filter
    :param session_key: string
        column name of the session key
    :param minlen: int
        minimum legnth required for a session to be kept
    :return: the filtered dataset
    """
    return df.join(
        df.groupBy(session_key).count().filter(F.col('count') >= minlen).select(session_key),
        on=session_key, how='inner')


if __name__ == '__main__':
    INPUT_PATH = './data/olx/raw'
    OUTPUT_PATH = './data/olx/clicks/clean'

    # Initialize Spark Session
    spark = SparkSession.builder \
        .master('local') \
        .appName('Click Prediction') \
        .config('spark.executor.memory', '5g') \
        .config('spark.driver.memory', '5g') \
        .getOrCreate()

    df_impr = spark.read.parquet(INPUT_PATH)

    # Keep only click events, session_len > 1, item_supp >= 10
    # df_impr = filter_by_value(df_impr, 'funnel', 1)  # keep clicks
    # df_impr = filter_by_item_support(df_impr, 'itemId', 10)  # keep frequent items
    # df_impr = filter_by_session_length(df_impr, 'search_id', 2)  # keep sessions with at least 2 actions

    # Keep only clicked events from 'search' channel.
    # df_impr = filter_by_value(df_impr, 'resultSetType', 'search')
    df_impr = filter_by_value(df_impr, 'funnel', 1)
    df_impr = filter_by_session_length(df_impr, 'search_id', 2)  # keep sessions with at least 2 actions
    df_impr = filter_by_item_support(df_impr, 'itemId', 10)  # keep frequent items

    df_impr.write.parquet(OUTPUT_PATH)
