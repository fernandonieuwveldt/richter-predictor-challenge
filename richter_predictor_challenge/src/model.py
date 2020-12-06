import tensorflow as tf

from richter_predictor_challenge.src.feature_encoder import NumericalFeatureEncoder, EmbeddingFeatureEncoder,  CategoricalFeatureEncoder
from richter_predictor_challenge.src.constants import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, HIGH_CARDINAL_FEATURES


class FeatureTransformer:
    """
    Feature encoder specifically for Wide and Deep network
    """
    def __init__(self):
        pass

    def transform(self, X):
        numerical_inputs, numerical_feature_encoders = NumericalFeatureEncoder(NUMERICAL_FEATURES).encode(X) 
        binary_inputs, binary_feature_encoders = CategoricalFeatureEncoder(BINARY_FEATURES).encode(X)
        categorical_inputs, categorical_feature_encoders = CategoricalFeatureEncoder(CATEGORICAL_FEATURES).encode(X)
        embedding_inputs, embedding_feature_encoders = EmbeddingFeatureEncoder(HIGH_CARDINAL_FEATURES).encode(X)
        high_categorical_inputs, high_categorical_feature_encoders = CategoricalFeatureEncoder(HIGH_CARDINAL_FEATURES).encode(X)

        feature_layer_inputs = {**numerical_inputs,
                                **binary_inputs,
                                **categorical_inputs,
                                **high_categorical_inputs,
                                **embedding_inputs
                                }

        feature_columns_wide = []
        feature_columns_wide.extend(binary_feature_encoders)
        feature_columns_wide.extend(categorical_feature_encoders)
        feature_columns_wide.extend(high_categorical_feature_encoders)

        feature_columns_deep = []
        feature_columns_deep.extend(embedding_feature_encoders)
        feature_columns_deep.extend(numerical_feature_encoders)        
        return feature_layer_inputs, feature_columns_wide, feature_columns_deep


def deep_classifier_v2(inputs, linear_feature_columns, dnn_feature_columns, dnn_hidden_units, total_classes):
    metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
               tf.keras.metrics.AUC(name='auc')]

    deep = tf.keras.layers.DenseFeatures(dnn_feature_columns)(inputs)
    for numnodes in dnn_hidden_units:
        deep = tf.keras.layers.Dense(numnodes, activation='relu')(deep)
        deep = tf.keras.layers.BatchNormalization()(deep)
        deep = tf.keras.layers.Dropout(0.25)(deep)
    wide = tf.keras.layers.DenseFeatures(linear_feature_columns)(inputs)
    both = tf.keras.layers.concatenate([deep, wide])
    output = tf.keras.layers.Dense(total_classes, activation='sigmoid')(both)
    model = tf.keras.Model(inputs=[v for v in inputs.values()], outputs=output)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0),
        optimizer=tf.keras.optimizers.RMSprop(lr=0.0005),
        metrics=metrics)
    return model


def deep_classifier(inputs, linear_feature_columns, dnn_feature_columns, dnn_hidden_units, total_classes):
    metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
               tf.keras.metrics.AUC(name='auc')]

    deep = tf.keras.layers.DenseFeatures(dnn_feature_columns)(inputs)
    wide = tf.keras.layers.DenseFeatures(linear_feature_columns)(inputs)
    both = tf.keras.layers.concatenate([deep, wide])
    for numnodes in dnn_hidden_units:
        both = tf.keras.layers.Dense(numnodes, activation='relu')(both)
        both = tf.keras.layers.BatchNormalization()(both)
        both = tf.keras.layers.Dropout(0.25)(both)
    output = tf.keras.layers.Dense(total_classes, activation='sigmoid')(both)
    model = tf.keras.Model(inputs=[v for v in inputs.values()], outputs=output)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0),
        optimizer=tf.keras.optimizers.RMSprop(lr=0.0005),
        metrics=metrics)
    return model

