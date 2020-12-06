
"""
workflow for training and predicting
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas
import numpy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from richter_predictor_challenge.src.data import TFDataTransformer
from richter_predictor_challenge.src.preprocessor import Preprocessor, DataSplitter
from richter_predictor_challenge.src.feature_encoder import CategoricalFeatureEncoder, EmbeddingFeatureEncoder, NumericalFeatureEncoder
from richter_predictor_challenge.src.constants import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, HIGH_CARDINAL_FEATURES
from richter_predictor_challenge.src.model import FeatureTransformer
from richter_predictor_challenge.src.model import deep_classifier_v2


EPOCHS = 100
BATCH_SIZE = 1024
train_data_features = pandas.read_csv("richter_predictor_challenge/data/train_values.csv").drop('building_id', axis=1)
test_data_features = pandas.read_csv("richter_predictor_challenge/data/test_values.csv")
raw_labels = pandas.read_csv("richter_predictor_challenge/data/train_labels.csv")

train_data_features = Preprocessor().transform(train_data_features)
test_data_features = Preprocessor().transform(test_data_features)

X_train, X_test, y_train, y_test = train_test_split(train_data_features, raw_labels)

test_data_features_copy = test_data_features.copy()
test_data_features = test_data_features.drop('building_id', axis=1)

def normalize(data_frame=None):
    return numpy.mean(data_frame[NUMERICAL_FEATURES]),\
           numpy.std(data_frame[NUMERICAL_FEATURES])

(X_train_no_damage, y_train_no_damage), (X_train_has_damage, y_train_has_damage) = DataSplitter().split(X_train, y_train)
(X_test_no_damage, y_test_no_damage), (X_test_has_damage, y_test_has_damage) = DataSplitter().split(X_test, y_test)

# no damage
# no_damage_mean, no_damage_std = normalize(X_train_no_damage)
# X_train_no_damage[NUMERICAL_FEATURES] = (X_train_no_damage[NUMERICAL_FEATURES] - no_damage_mean) / no_damage_std
# X_test_no_damage[NUMERICAL_FEATURES]  = (X_test_no_damage[NUMERICAL_FEATURES] - no_damage_mean) / no_damage_std

# has damage
# has_damage_mean, has_damage_std = normalize(X_train_has_damage)
# X_train_has_damage[NUMERICAL_FEATURES] = (X_train_has_damage[NUMERICAL_FEATURES] - has_damage_mean) / has_damage_std
# X_test_has_damage[NUMERICAL_FEATURES]  = (X_test_has_damage[NUMERICAL_FEATURES] - has_damage_mean) / has_damage_std

# No damage dataset
train_dataset_no_damage = TFDataTransformer().transform(X_train_no_damage, y_train_no_damage).batch(BATCH_SIZE)
val_dataset_no_damage = TFDataTransformer().transform(X_test_no_damage, y_test_no_damage).batch(BATCH_SIZE)

# Has damage dataset
train_dataset_has_damage = TFDataTransformer().transform(X_train_has_damage, y_train_has_damage).batch(BATCH_SIZE)
val_dataset_has_damage = TFDataTransformer().transform(X_test_has_damage, y_test_has_damage).batch(BATCH_SIZE)


feature_layer_inputs, feature_columns_wide, feature_columns_deep = FeatureTransformer().transform(X_train)


model_no_damage = deep_classifier_v2(feature_layer_inputs, feature_columns_wide, feature_columns_deep, [512, 64], total_classes=1)

early_stopping = tf.keras.callbacks.EarlyStopping(**{'monitor': 'val_loss',
                                                     'mode': 'min',
                                                     'verbose': 1,
                                                     'patience': 10})


model_checkpoint = tf.keras.callbacks.ModelCheckpoint(**{'filepath': '/tmp/best_model_no_damage',
                                                         'monitor': 'val_loss',
                                                         'mode': 'min',
                                                         'verbose': 1,
                                                         'save_weights_only': True,
                                                         'save_best_only': True})

class_weights = class_weight.compute_class_weight('balanced', numpy.unique(y_train_no_damage.values), y_test_no_damage.values.flatten())
class_weights = dict(enumerate(class_weights))

model_no_damage.fit(train_dataset_no_damage,
                    epochs=EPOCHS,
                    validation_data=val_dataset_no_damage,
                    callbacks=[early_stopping, model_checkpoint],
                    class_weight=class_weights
                    )


model_no_damage.load_weights('/tmp/best_model_no_damage')
model_has_damage = deep_classifier_v2(feature_layer_inputs, feature_columns_wide, feature_columns_deep, [512, 64], total_classes=2)

early_stopping = tf.keras.callbacks.EarlyStopping(**{'monitor': 'val_loss',
                                                     'mode': 'min',
                                                     'verbose': 1,
                                                     'patience': 10})

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(**{'filepath': '/tmp/best_model_has_damage',
                                                         'monitor': 'val_loss',
                                                         'mode': 'min',
                                                         'verbose': 1,
                                                         'save_weights_only': True,
                                                         'save_best_only': True})
model_has_damage.fit(train_dataset_has_damage,
                     epochs=EPOCHS,
                     validation_data=val_dataset_has_damage,
                     callbacks=[early_stopping, model_checkpoint]
          )

model_has_damage.load_weights('/tmp/best_model_has_damage')


validation_probas = model_no_damage.predict(val_dataset_no_damage)


def calibrate(target, confidence):
    prob_thres = numpy.arange(0, 1.01, 0.01)
    accuracy_threshold = 0.0 * prob_thres
    for idx, thres in enumerate(prob_thres):
        metric = tf.keras.metrics.BinaryAccuracy(threshold=thres)
        metric.reset_states()
        metric.update_state(target, confidence)
        accuracy_threshold[idx] = metric.result().numpy()
    return prob_thres[numpy.argmax(accuracy_threshold)]


import sklearn.metrics
def optimal_threshold_precision_recall(target, confidence):
    p, r, thresholds = sklearn.metrics.precision_recall_curve(target, confidence)
    fscore = (2 * p * r) / (p+r)
    idx = numpy.argmax(fscore)
    return thresholds[idx]


CALIBRATION_THRESHOLD =  0.5  # optimal_threshold_precision_recall(y_test_no_damage.values, validation_probas)

# Apply models
# test_data_features[NUMERICAL_FEATURES] = (test_data_features_copy[NUMERICAL_FEATURES] - no_damage_mean) / no_damage_std 
test_dataset = tf.data.Dataset.from_tensor_slices(dict(test_data_features)).batch(BATCH_SIZE)
pred_no_damage_probas = model_no_damage.predict(test_dataset)
pred_no_damage = 1.0 * (pred_no_damage_probas >= CALIBRATION_THRESHOLD)


test_data_features_has_damage = test_data_features.loc[pred_no_damage==0, :]
# test_data_features_has_damage[NUMERICAL_FEATURES] = (test_data_features_has_damage[NUMERICAL_FEATURES] - has_damage_mean) / has_damage_std
test_dataset_has_damage = tf.data.Dataset.from_tensor_slices(dict(test_data_features_has_damage)).batch(BATCH_SIZE)
pred_has_damage = model_has_damage.predict(test_dataset_has_damage).argmax(axis=1)+2

pred_no_damage[pred_no_damage==0]=pred_has_damage


def submit_run(file_name=None):
    submit_data_frame = pandas.DataFrame([], columns=['building_id', 'damage_grade'])
    submit_data_frame['building_id'] = test_data_features_copy['building_id'].values
    submit_data_frame['damage_grade'] = pred_no_damage
    submit_data_frame['damage_grade'] = submit_data_frame['damage_grade'].map(int)
    submit_data_frame.to_csv(f'{file_name}.csv', index=False)


submit_run('results')
