# Implementation of a simple MLP network with one hidden layer

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from processData import load_data_and_labels
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h = tf.nn.sigmoid(tf.matmul(X, w_1))
    yhat = tf.matmul(h, w_2)
    return yhat

def get_data():

  # Load data
  x_text, y = load_data_and_labels("email_contents.npy", "labels.npy")
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
  return train_test_split(x, y, test_size=0.3, random_state=42)

def plotAccuracyVsTime(num_epochs, train_accuracies, test_accuracies, filename):
  x_values = [i + 1 for i in range(num_epochs)]
  plt.plot(x_values, train_accuracies)
  plt.plot(x_values, test_accuracies)
  plt.xlabel("epoch")
  plt.ylabel("accuracy")
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig(filename)

def main():
    train_X, test_X, train_y, test_y = get_data()

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

    # Forward propagation
    yhat = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    train_accuracies = []
    test_accuracies = []
    num_epochs = 2
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