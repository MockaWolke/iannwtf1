"""
Preprocessing script
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def onehotify(tensor):
    vocab = {'A':'1', 'C':'2', 'G':'3', 'T':'0'}
    for key in vocab.keys():
        tensor = tf.strings.regex_replace(tensor, key, vocab[key])
    split = tf.strings.bytes_split(tensor)
    labels = tf.cast(tf.strings.to_number(split), tf.uint8)
    onehot = tf.one_hot(labels, 4)
    onehot = tf.reshape(onehot, (-1,))
    return onehot


def prep_data(ds_train, ds_test):
    new_ds_train = ds_train.map(lambda input, target: (onehotify(input),tf.one_hot(target,10))) # map input to onehotify and targets to onehot vectors
    new_ds_test = ds_test.map(lambda input, target: (onehotify(input),tf.one_hot(target,10))) # map input to onehotify and targets to onehot vectors
    new_ds_train = new_ds_train.shuffle(10000).batch(1000).prefetch(tf.data.AUTOTUNE)
    new_ds_test = new_ds_test.shuffle(100).batch(100).prefetch(tf.data.AUTOTUNE)
    return new_ds_train, new_ds_test