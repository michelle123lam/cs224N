# Implementation of a simple one-layer neural network

from __future__ import print_function

import time
import datetime
import os
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
from sklearn.model_selection import train_test_split
from processData import load_data_and_labels, load_data_and_labels_bow, batch_iter
from utils.treebank import StanfordSentiment
import utils.glove as glove

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
    sentVector = np.zeros((wordVectors.shape[1],))

    indices = []
    for word in sentence:
        if tokens.get(word, 0) == 0:
            pass
            #print "this word %s does not appear in the glove vector initialization" % word
        else:
            #print "this word %s DOES appear in the glove vector initialization" % word
            indices.append(tokens[word])
    if len(indices) == 0:
      return None

    sentVector = np.mean(wordVectors[indices, :], axis=0)

    assert sentVector.shape == (wordVectors.shape[1],)
    return sentVector

def get_glove_data():
  embedding_dimension = 100
  x_text, y = load_data_and_labels("email_contents.npy", "labels.npy")
  num_recipients_features = np.array(np.load("num_recipients_features.npy"))

  dataset = StanfordSentiment()
  tokens = dataset.tokens()
  nWords = len(tokens)

  # Initialize word vectors with glove.
  embedded_vectors = glove.loadWordVectors(tokens)
  print("The shape of embedding matrix is:")
  print(embedded_vectors.shape) # Should be number of e-mails, number of embeddings

  nTrain = len(x_text)
  trainFeatures = np.zeros((nTrain, embedding_dimension + 2)) #5 is the number of slots the extra features take up
  toRemove = []
  for i in xrange(nTrain):
    words = x_text[i]
    num_words = len(words)

    #place number of words in buckets
    if num_words < 10:
        num_words_bucket = 0
    elif num_words >= 10 and num_words < 100:
        num_words_bucket = 1
    elif num_words >= 100 and num_words < 500:
        num_words_bucket = 2
    elif num_words >= 500 and num_words < 1000:
        num_words_bucket = 3
    elif num_words >= 1000 and num_words < 2000:
        num_words_bucket = 4
    elif num_words >= 2000:
        num_words_bucket = 5

    sentenceFeatures = getSentenceFeatures(tokens, embedded_vectors, words)
    if sentenceFeatures is None:
      toRemove.append(i)
    else:
      featureVector = np.hstack((sentenceFeatures, num_recipients_features[i]))
      featureVector = np.hstack((featureVector, num_words_bucket))
      trainFeatures[i, :] = featureVector

  y = np.delete(y, toRemove, axis=0)
  trainFeatures = np.delete(trainFeatures, toRemove, axis=0)

  # Randomly shuffle data
  np.random.seed(10)
  shuffle_indices = np.random.permutation(np.arange(len(y)))  # Array of random numbers from 1 to # of labels.
  x_shuffled = trainFeatures[shuffle_indices]
  y_shuffled = y[shuffle_indices]

  train = 0.7
  dev = 0.3
  return train_test_split(x_shuffled, y_shuffled, test_size=0.3, random_state=42)

def get_count_data():

  # Load data
  x_text, y = load_data_and_labels_bow("email_contents.npy", "labels.npy")

  # Reminder: dominant sender, subordinate recipient equals label 0

  # Build vocabulary
  max_email_length = max([len(x.split(" ")) for x in x_text])
  # Function that maps each email to sequences of word ids. Shorter emails will be padded.
  vocab_processor = learn.preprocessing.VocabularyProcessor(max_email_length)
  # x is a matrix where each row contains a vector of integers corresponding to a word.
  x = np.array(list(vocab_processor.fit_transform(x_text)))

  # Randomly shuffle data
  np.random.seed(10)
  shuffle_indices = np.random.permutation(np.arange(len(y)))  # Array of random numbers from 1 to # of labels.
  x_shuffled = x[shuffle_indices]
  y_shuffled = y[shuffle_indices]

  train = 0.7
  dev = 0.3
  return train_test_split(x_shuffled, y_shuffled, test_size=0.3, random_state=42)

train_X, test_X, train_y, test_y = get_glove_data()

# Parameters
RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)
learning_rate = 0.01
training_epochs = 100
batch_size = 500
display_step = 1
evaluate_every = 50

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_input = train_X.shape[1] # MNIST data input (img shape: 28*28)
n_classes = train_y.shape[1] # MNIST total classes (0-9 digits)

global_step = tf.Variable(0, name="global_step", trainable=False)
# tf Graph Input
x = tf.placeholder(tf.float32, [None, n_input], name='X')
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, n_classes], name='y')

# Create model
def perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.softmax(layer_1)
    # Create a summary to visualize the first layer ReLU activation
    tf.summary.histogram("softmax1", layer_1)
    # Output layer
    out_layer = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    return out_layer

# Store layers weight & bias
weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='W1'),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_classes]), name='W2'),
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
    'b2': tf.Variable(tf.random_normal([n_classes]), name='b2'),
}

# Encapsulating all ops into scopes, making Tensorboard's Graph
# Visualization more convenient
pred = perceptron(x, weights, biases)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# Op to calculate every variable gradient
grads = tf.gradients(loss, tf.trainable_variables())
grads = list(zip(grads, tf.trainable_variables()))
# Op to update all variables according to their gradient
apply_grads = optimizer.apply_gradients(grads_and_vars=grads, global_step=global_step)

acc = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "oneLayer_runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Create a summary to monitor cost tensor
    loss_summary = tf.summary.scalar("loss", loss)
    # Create a summary to monitor accuracy tensor
    acc_summary = tf.summary.scalar("accuracy", acc)

    # Summarize all gradients
    grad_summaries = []
    for g, v in grads:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Generate batches
    train_batches = batch_iter(
        list(zip(train_X, train_y)), batch_size, training_epochs)
    dev_batches = batch_iter(
        list(zip(test_X, test_y)), batch_size, training_epochs)

    # Training cycle
    for train_batch, dev_batch in zip(train_batches, dev_batches):
        x_train_batch, y_train_batch = zip(*train_batch)
        _, step, summaries, loss_1, accuracy = sess.run([apply_grads, global_step, train_summary_op, loss, acc],
                                     feed_dict={x: x_train_batch, y: y_train_batch})
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss_1, accuracy))
        train_summary_writer.add_summary(summaries, step)
        train_summary_writer.flush()

        current_step = tf.train.global_step(sess, global_step)
        if current_step % evaluate_every == 0:
            print("\nEvaluation:")
            x_dev_batch, y_dev_batch = zip(*dev_batch)
            step, summaries, loss_1, accuracy = sess.run([global_step, dev_summary_op, loss, acc],
                                     feed_dict={x: x_dev_batch, y: y_dev_batch})
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss_1, accuracy))
            dev_summary_writer.add_summary(summaries, step)
            dev_summary_writer.flush()
            print("")

    train_summary_writer.close()
    dev_summary_writer.close()