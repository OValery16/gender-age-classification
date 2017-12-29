from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os
import numpy as np
import tensorflow as tf
#from data import distorted_inputs
import re
from tensorflow.contrib.layers import *

def genderClassifier():

    train_inputs=2048
    train_labels=2


    # Setting hyperparameters
    learning_rate = 0.01
    batch_size = 64
    epochs = 50
    log_batch_step = 50

    # useful info
    n_features = 2048
    n_labels = 2

    # Placeholders for input features and labels
    inputs = tf.placeholder(tf.float32, (None, n_features))
    labels = tf.placeholder(tf.float32, (None, n_labels))

    # Setting up weights and bias
    weights = tf.Variable(tf.truncated_normal((n_features, n_labels), stddev=0.1), name='weights')
    bias = tf.Variable(tf.zeros(n_labels), name='bias')
    tf.add_to_collection('vars', weights)
    tf.add_to_collection('vars', bias)

    # Setting up operation in fully connected layer
    logits = tf.add(tf.matmul(inputs, weights), bias)
    prediction = tf.nn.softmax(logits)
    tf.add_to_collection('pred', prediction)

    # Defining loss of network
    difference = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_sum(difference)

    # Setting optimiser
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Define accuracy
    is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

    saver = tf.train.Saver((weights, bias))

    return prediction,inputs
