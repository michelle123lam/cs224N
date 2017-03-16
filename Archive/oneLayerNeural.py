# Implementation of a simple MLP network with one hidden layer

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from processData import load_data_and_labels, load_data_and_labels_bow, load_embedding_vectors_glove
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from utils.treebank import StanfordSentiment
import utils.glove as glove
import createFeatVecs

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2, b_1, b_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h = tf.nn.sigmoid(tf.matmul(X, w_1) + b_1)
    yhat = tf.matmul(h, w_2) + b_2
    return yhat

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
  print "The shape of embedding matrix is:"
  print embedded_vectors.shape  # Should be number of e-mails, number of embeddings

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
        num_words_buckets = 2
    elif num_words >= 500 and num_words < 1000:
        num_words_buckets = 3
    elif num_words >= 1000 and num_words < 2000:
        num_words_buckets = 4
    elif num_words >= 2000:
        num_words_buckets = 5

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

def plotAccuracyVsTime(num_epochs, train_accuracies, test_accuracies, filename):
  x_values = [i + 1 for i in range(num_epochs)]
  plt.plot(x_values, train_accuracies)
  plt.plot(x_values, test_accuracies)
  plt.xlabel("epoch")
  plt.ylabel("accuracy")
  #plt.ylim(ymin=0)
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig(filename)

def main():
    train_X, test_X, train_y, test_y = get_glove_data()

    # Layer's sizes
    x_size = train_X.shape[1]
    h_size = 100

    y_size = train_y.shape[1]

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    b_1 = tf.Variable(tf.zeros([h_size]))
    b_2 = tf.Variable(tf.zeros([y_size]))

    # Forward propagation
    yhat = forwardprop(X, w_1, w_2, b_1, b_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.005).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    train_accuracies = []
    test_accuracies = []
    num_epochs = 30
    for epoch in range(num_epochs):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))
        train_accuracies.append(train_accuracy)
        test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: test_X, y: test_y}))
        test_accuracies.append(test_accuracy)

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    sess.close()
    plotAccuracyVsTime(num_epochs, train_accuracies, test_accuracies, "oneLayerNeuralPlot.png")

if __name__ == '__main__':
    main()