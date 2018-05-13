#Implementation of SVM baseline

from __future__ import print_function

import time
import datetime
import os
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from processData import load_data_and_labels, load_data_and_labels_bow, batch_iter
from utils.treebank import StanfordSentiment
import utils.glove as glove
from sklearn import metrics
import nltk
import oneLayerNeural

def single_svm():

    x_text, y = load_data_and_labels_bow("email_contents_nodup.npy", "labels_nodup.npy")

    # Randomly shuffle data
    shuffle_indices = list(np.random.permutation(np.arange(len(y))))  # Array of random numbers from 1 to # of labels.
    x_shuffled = [x_text[index] for index in shuffle_indices]
    y_shuffled = [y[index] for index in shuffle_indices]

    train = 0.6
    dev = 0.2
    test = 0.2
    # train x, dev x, test x, train y, dev y, test y
    train_cutoff = int(0.6 * len(x_shuffled))
    dev_cutoff = int(0.8 * len(x_shuffled))
    test_cutoff = int(len(x_shuffled))

    train_X = x_shuffled[0:train_cutoff]
    dev_X = x_shuffled[train_cutoff:dev_cutoff]
    test_X = x_shuffled[dev_cutoff:test_cutoff]
    train_y = y_shuffled[0:train_cutoff]
    dev_y = y_shuffled[train_cutoff:dev_cutoff]
    test_y = y_shuffled[dev_cutoff:test_cutoff]

    train_y_single = []
    for i, label in enumerate(train_y):
        if (label[0] == 0):
            train_y_single.append(1)
        if (label[0] == 1):
            train_y_single.append(0)

    test_y_single = []
    for i, label in enumerate(test_y):
        if (label[0] == 0):
            test_y_single.append(1)
        if (label[0] == 1):
            test_y_single.append(0)

    text_clf = Pipeline([('vect', CountVectorizer()),
                    ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)), ])
    text_clf.fit(np.asarray(train_X), np.asarray(train_y_single))
    predicted = text_clf.predict(np.asarray(test_X))
    print(np.mean(predicted == test_y_single))
    print(metrics.classification_report(np.asarray(test_y_single), predicted))

def grouped_svm():
    pass

single_svm()
