"""
RNN using LSTM cells based on TensorFlow tutorial
Michelle Lam - Winter 2017
"""

import numpy as np
import random
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
from processData import batch_iter, load_data_and_labels, load_embedding_vectors_glove
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from utils.treebank import StanfordSentiment
import utils.glove as glove

def plotAccuracyVsTime(num_epochs, train_accuracies, test_accuracies, filename, y_var_name):
  x_values = [i + 1 for i in range(num_epochs)]
  plt.figure()
  plt.plot(x_values, train_accuracies)
  plt.plot(x_values, test_accuracies)
  plt.xlabel("epoch")
  plt.ylabel(y_var_name)
  plt.ylim(ymin=0)
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig(filename)


def getSentenceFeatures(tokens, wordVectors, sentence):
    """
    Obtain the sentence feature for sentiment analysis by averaging its
    word vectors
    """

    # Implement computation for the sentence features given a sentence.

    # Inputs:
    # tokens -- a dictionary that maps words to their indices in
    #           the word vector list
    # wordVectors -- word vectors (each row) for all tokens
    # sentence -- a list of words in the sentence of interest

    # Output:
    # - sentVector: feature vector for the sentence
    # sentVector = np.zeros((wordVectors.shape[1],))

    sentVector = []
    # indices = []
    for word in sentence:
        if tokens.get(word, 0) == 0:
            print "this word %s does not appear in the glove vector initialization" % word
        else:
            # indices.append(tokens[word])
            sentVector.append(wordVectors[tokens[word]])

    # sentVector = np.mean(wordVectors[indices, :], axis=0)
    sentVector = np.array(sentVector)
    print sentVector.shape

    # assert sentVector.shape == (wordVectors.shape[1],)
    return sentVector


# ==================================================
# LOAD DATA

emails, labels = load_data_and_labels("email_contents.npy", "labels.npy")
# emails, labels = load_data_and_labels("email_contents_grouped.npy", "labels_grouped.npy")
emails = np.array(emails)
print "emails shape:", emails.shape

max_email_length = max([len(email) for email in emails])
print "The max_email_length is %d" % max_email_length

dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)
embedding_dim = 100

# Initialize word vectors with glove.
wordVectors = glove.loadWordVectors(tokens)
print "The shape of embedding matrix is:"
print wordVectors.shape  # Should be number of e-mails, number of embeddings

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(labels)))  # Array of random numbers from 1 to # of labels.
emails_shuffled = emails[shuffle_indices]
labels_shuffled = labels[shuffle_indices]

train = 0.7
dev = 0.3
x_train, x_test, y_train, y_test = train_test_split(emails_shuffled, labels_shuffled, test_size=0.3, random_state=42)
print "x_train", x_train.shape
print "x_test", x_test.shape
print "y_train", y_train.shape
print "y_test", y_test.shape

zeros = np.zeros((wordVectors.shape[1],))
# Load train set and initialize with glove vectors.
nTrain = len(x_train)
trainFeatures = np.zeros((nTrain, embedding_dim))  # dimVectors should be embedding_dim
trainLabels = y_train
for i in xrange(nTrain):
    words = x_train[i]
    trainFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

# Prepare test set features
nTest = len(x_test)
testFeatures = np.zeros((nTest, embedding_dim))
testLabels = y_test
for i in xrange(nTest):
    words = x_test[i]
    testFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

print "trainFeatures", trainFeatures.shape
print "trainLabels", trainLabels.shape
print "testFeatures", testFeatures.shape
print "testLabels", testLabels.shape


# ==================================================
# LSTM

NUM_EXAMPLES = trainFeatures.shape[0] # 47411
RNN_HIDDEN = 60
LEARNING_RATE = 0.01

BATCH_SIZE = 1000
N_TIMESTEPS = max_email_length # 1
N_CLASSES = 2
N_INPUT = trainFeatures.shape[1] # 14080
keep_rate = 0.5

# Account for 1 timestep
# x_train = np.expand_dims(x_train, axis=1)
# x_test = np.expand_dims(x_test, axis=1)

data = tf.placeholder(tf.float32, [None, N_TIMESTEPS, N_INPUT]) # (batch_size, n_timesteps, n_features)
target = tf.placeholder(tf.float32, [None, N_CLASSES])

cell = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN)
# cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=keep_rate, output_keep_prob=keep_rate)
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
print "rnn_outputs1", rnn_outputs.get_shape()
# rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])
rnn_outputs = tf.reshape(rnn_outputs, [-1, RNN_HIDDEN])
print "rnn_outputs2", rnn_outputs.get_shape()

weight = tf.Variable(tf.truncated_normal([RNN_HIDDEN, N_CLASSES]))
# weight = tf.get_variable("weight", shape=[RNN_HIDDEN, N_CLASSES], initializer=tf.contrib.layers.xavier_initializer())
bias = tf.Variable(tf.constant(0.1, shape=[N_CLASSES]))
print "weight", weight.get_shape()
print "bias", bias.get_shape()

prediction = tf.nn.softmax(tf.matmul(rnn_outputs, weight) + bias)
print "prediction", prediction.get_shape()
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=target))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
minimize = optimizer.minimize(cross_entropy)

y_p = tf.argmax(prediction, 1)
y_t = tf.argmax(target, 1)

# Evaluate model
correct_pred = tf.equal(y_t, y_p)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Execution of the graph
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)


# ==================================================
# RUN LSTM EPOCHS

no_of_batches = int(NUM_EXAMPLES/BATCH_SIZE)
n_epochs = 10

train_accuracies = []
train_losses = []
test_accuracies = []
test_losses = []

for i in range(n_epochs):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = trainFeatures[ptr:ptr+BATCH_SIZE], trainLabels[ptr:ptr+BATCH_SIZE]
        ptr+=BATCH_SIZE
        sess.run(minimize ,{data: inp, target: out})

        if j % no_of_batches == 0:
          # Calculate batch accuracy and loss
          acc, loss = sess.run([accuracy, cross_entropy] , {data: inp, target: out})
          train_accuracies.append(acc)
          train_losses.append(loss)
          print "Iter " + str(i*BATCH_SIZE) + ", Minibatch Loss= " + \
            "{:.6f}".format(loss) + ", Training Accuracy= " + \
            "{:.5f}".format(acc)

          # Calculate test accuracy and loss
          acc, loss = sess.run([accuracy, cross_entropy], {data: testFeatures, target: testLabels})
          test_accuracies.append(acc)
          test_losses.append(loss)
          print "Testing Loss= " + "{:.6f}".format(loss) + \
                ", Testing Accuracy= " + "{:.5f}".format(acc)


# Final evaluation of test accuracy and loss
acc, loss, y_pred, y_target = sess.run([accuracy, cross_entropy, y_p, y_t], {data: testFeatures, target: testLabels})
print "Testing Loss= " + "{:.6f}".format(loss) + \
      ", Testing Accuracy= " + "{:.5f}".format(acc)

# Plot accuracies, losses over time
plotAccuracyVsTime(n_epochs, train_accuracies, test_accuracies, "LSTMAccuracyPlot.png", "accuracy")
plotAccuracyVsTime(n_epochs, train_losses, test_losses, "LSTMLossPlot.png", "cross-entropy loss")

# Report precision, recall, F1
print "Precision: {}%".format(100*metrics.precision_score(y_target, y_pred, average="weighted"))
print "Recall: {}%".format(100*metrics.recall_score(y_target, y_pred, average="weighted"))
print "f1_score: {}%".format(100*metrics.f1_score(y_target, y_pred, average="weighted"))



sess.close()

