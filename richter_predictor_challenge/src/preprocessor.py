
"""
preprocess raw data
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas
import numpy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from richter_predictor_challenge.src.data import TFDataTransformer
from richter_predictor_challenge.src.constants import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, HIGH_CARDINAL_FEATURES


class Preprocessor:

    def __init__(self):
        pass

    def transform(self, X, y=None):
        # X['age'] = X['age'].map(lambda age: min(age, 30))
        # X['height_percentage'] = X['height_percentage'].map(lambda height: min(height, 11))
        # X['area_percentage'] = X['area_percentage'].map(lambda area: min(area, 23))
        # X['count_floors_pre_eq'] = X['count_floors_pre_eq'].map(lambda floors: min(floors, 6))
        X['area_by_height'] = X['area_percentage'] * X['height_percentage']
        # set data types
        X = X.astype('str')
        X[NUMERICAL_FEATURES] = X[NUMERICAL_FEATURES].astype('float32')
        return X


class DataSplitter:
    def __init__(self):
        pass

    def transform_labels(self, labels=None):
        """
        Transform raw label dataset into categorical matrix
        """
        encoded_labels = pandas.DataFrame([], columns=['low', 'medium', 'high'])
        encoded_labels.loc[:, 'low'] = 1.0 * (labels['damage_grade'].values == 1)
        encoded_labels.loc[:, 'medium'] = 1.0 * (labels['damage_grade'].values == 2)
        encoded_labels.loc[:, 'high'] = 1.0 * (labels['damage_grade'].values == 3)
        return encoded_labels

    def extract_low_damage(self, X, y):
        return X, y.loc[:, 'low']
        # return X, y.loc[:, 'high']

    def extract_medium_damage(self, X, y):
        X = X.reset_index().drop('index', axis=1)
        has_damage_index = y.loc[:, 'low'] == 0
        y_ = y.loc[has_damage_index, ['medium', 'high']]
        # has_damage_index = y.loc[:, 'high'] == 0
        # y_ = y.loc[has_damage_index, ['low', 'medium']]
        X_ = X.loc[has_damage_index, :]
        return X_, y_

    def split(self, X, y):
        encoded_labels = self.transform_labels(y)
        return self.extract_low_damage(X, encoded_labels), self.extract_medium_damage(X, encoded_labels)
