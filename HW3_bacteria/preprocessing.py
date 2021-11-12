"""
Preprocessing script
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def onehotify(tensor):
    """One-Hot encoding for our dataset."""
    vocab = {'A':'1', 'C':'2', 'G':'3', 'T':'0'}
    # for loop to replace the respective letter with a number
    for key in vocab.keys():
        tensor = tf.strings.regex_replace(tensor, key, vocab[key])
    split = tf.strings.bytes_split(tensor)
    labels = tf.cast(tf.strings.to_number(split), tf.uint8)
    onehot = tf.one_hot(labels, 4)
    onehot = tf.reshape(onehot, (-1,))
    return onehot


def prep_data(ds_train, ds_test):
    """Applies one-hot encoding on our dataset."""
    # map input to onehotify and targets to onehot vectors
    new_ds_train = ds_train.map(lambda input, target: (onehotify(input),tf.one_hot(target,10)))
    # map input to onehotify and targets to onehot vectors
    new_ds_test = ds_test.map(lambda input, target: (onehotify(input),tf.one_hot(target,10)))
    # different batch and shuffle sizes are useful for our train and test dataset
    new_ds_train = new_ds_train.shuffle(10000).batch(1000).prefetch(tf.data.AUTOTUNE) # tf.data.AUTOTUNE tunes our prefetch-size
    new_ds_test = new_ds_test.shuffle(100).batch(100).prefetch(tf.data.AUTOTUNE) # dynamically at runtime to make it as fast as possible 
    return new_ds_train, new_ds_test