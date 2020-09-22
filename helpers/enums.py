""" 
Created at 27/05/2019

@author: dimitris.michailidis
"""

from enum import IntEnum


class LatentMode(IntEnum):
    NONE = 1
    # Appends location information after the hidden state
    LATENTAPPEND = 2
    # Learns a latent representation in the embedding level, fuses it to the event embedding.
    LATENTEMBEDD = 3
    # Multiplies latent location representation after recurrent unit.
    LATENTMULTIPLY = 4
    # Concatenates location information after recurrent unit.
    LATENTCONCAT = 5


class LocationMode(IntEnum):
    NONE = 1
    # Distance-one-hot encode: 0,0,0,X,0,0,0,0 where X = distance instead of 1
    DISTHOT = 2
    # The full location information including district, city, distance
    FULLCONCAT = 3
    # 1-hot encode of city
    CITYHOT = 4
    # 1-hot encode of district
    DISTRICTHOT = 5
    # Concatetnation of distance to the main event embedding (only distance)
    CONCAT = 6
    # Distance alone acts as the location representation.
    DISTANCE = 7


class ParallelMode(IntEnum):
    NONE = 1
    # Combines the two models on the encoder level
    ENCODER = 2
    # Combines the two models on the decoder level
    DECODER = 3
    # Combines the two models on the hidden state level
    HIDDEN = 4


class CombinationMode(IntEnum):
    # How the two parallel models are combined.
    # Concatenates the weights of the two models
    CONCAT = 1
    # Multiplies the weights of the two models
    MULTIPLY = 2
    # Adds the weights of the two models
    ADD = 3
    # Creates a weighed SUM of the weights of the two models
    WEIGHTED_SUM = 4
