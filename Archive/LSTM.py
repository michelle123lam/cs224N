"""
RNN using LSTM cells based on TensorFlow tutorial
Michelle Lam - Winter 2017
"""

import numpy as np
import random
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
from processData import batch_iter, load_data_and_labels

####################
# LOAD DATA

# Load data
x_text, y = load_data_and_labels("email_contents.npy", "labels.npy")
# dominant sender, subordinate recipient = label 0
print y[0]
print "y", y.shape

# Build vocabulary
max_email_length = max([len(x.split(" ")) for x in x_text])
# Function that maps each email to sequences of word ids. Shorter emails will be padded.
vocab_processor = learn.preprocessing.VocabularyProcessor(max_email_length)
# x is a matrix where each row contains a vector of integers corresponding to a word.
x = np.array(list(vocab_processor.fit_transform(x_text)))
print "x", x.shape

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))  # Array of random numbers from 1 to # of labels.
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

train = 0.7
dev = 0.3
x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.3, random_state=42)
x_train = np.expand_dims(x_train, axis=1)
x_dev = np.expand_dims(x_dev, axis=1)
print "x_train", x_train.shape
print "x_dev", x_dev.shape
print "y_train", y_train.shape
print "y_dev", y_dev.shape




## second version
NUM_EXAMPLES = 47411 # 67730
RNN_HIDDEN = 60
LEARNING_RATE = 1

BATCH_SIZE = 1000 #64
N_TIMESTEPS = 1
N_CLASSES = 2
N_INPUT = 14080

keep_rate = 0.5


data = tf.placeholder(tf.float32, [None, N_TIMESTEPS, N_INPUT]) # (batch_size, n_timesteps, n_features)
target = tf.placeholder(tf.float32, [None, N_CLASSES])

cell = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN)
# cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=keep_rate, output_keep_prob=keep_rate)
rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
print "rnn_outputs1", rnn_outputs.get_shape()
# rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])
rnn_outputs = tf.reshape(rnn_outputs, [-1, RNN_HIDDEN])
print "rnn_outputs2", rnn_outputs.get_shape()

weight = tf.Variable(tf.random_normal([RNN_HIDDEN, N_CLASSES], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[N_CLASSES]))
print "weight", weight.get_shape()
print "bias", bias.get_shape()

prediction = tf.nn.softmax(tf.matmul(rnn_outputs, weight) + bias)
print "prediction", prediction.get_shape()
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=target))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
minimize = optimizer.minimize(cross_entropy)

# Evaluate model
correct_pred = tf.equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Execution of the graph
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

no_of_batches = int(NUM_EXAMPLES/BATCH_SIZE)
n_epochs = 10 #200

for i in range(n_epochs):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = x_train[ptr:ptr+BATCH_SIZE], y_train[ptr:ptr+BATCH_SIZE]
        ptr+=BATCH_SIZE
        sess.run(minimize ,{data: inp, target: out})

        if j % no_of_batches == 0:
          # Calculate batch accuracy
          acc = sess.run(accuracy, {data: inp, target: out})
          # Calculate batch loss
          loss = sess.run(cross_entropy, {data: inp, target: out})
          print "Iter " + str(i*BATCH_SIZE) + ", Minibatch Loss= " + \
            "{:.6f}".format(loss) + ", Training Accuracy= " + \
            "{:.5f}".format(acc)

          acc = sess.run(accuracy, {data: x_dev, target: y_dev})
          loss = sess.run(cross_entropy, {data: x_dev, target: y_dev})
          print "Testing Loss= " + "{:.6f}".format(loss) + \
                ", Testing Accuracy= " + "{:.5f}".format(acc)

# x_dev = np.expand_dims(x_dev, axis=1)
# acc = sess.run(accuracy, {data: x_dev, target: y_dev})
# loss = sess.run(cross_entropy, {data: x_dev, target: y_dev})
# print "Testing Loss= " + "{:.6f}".format(loss) + \
#       ", Testing Accuracy= " + "{:.5f}".format(acc)
sess.close()
