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


# ==================================================
# PARAMETERS
# Model Hyperparameters
# tf.flags.DEFINE_boolean("enable_word_embeddings", True, "Enable/disable the word embedding (default: True)")

# FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

# # Setup for word embeddings
# with open("config.yml", 'r') as ymlfile:
#     cfg = yaml.load(ymlfile)

# if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
#     embedding_name = cfg['word_embeddings']['default']
#     embedding_dimension = cfg['word_embeddings'][embedding_name]['dimension']
# else:
#     embedding_dimension = FLAGS.embedding_dim


# ==================================================
# LOAD DATA

x_text, y = load_data_and_labels("email_contents.npy", "labels.npy")
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


# ==================================================
# LSTM

NUM_EXAMPLES = 47411
RNN_HIDDEN = 60
LEARNING_RATE = 0.01

BATCH_SIZE = 1000
N_TIMESTEPS = 1
N_CLASSES = 2
N_INPUT = 14080
keep_rate = 0.5

data = tf.placeholder(tf.float32, [None, N_TIMESTEPS, N_INPUT]) # (batch_size, n_timesteps, n_features)
target = tf.placeholder(tf.float32, [None, N_CLASSES])

cell = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN)
cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=keep_rate, output_keep_prob=keep_rate)
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

# Evaluate model
correct_pred = tf.equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Execution of the graph
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

# # Load word embeddings
# if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
#   vocabulary = vocab_processor.vocabulary_
#   initW = None
#   if embedding_name == 'word2vec':
#     # load embedding vectors from the word2vec
#     print("Load word2vec file {}".format(cfg['word_embeddings']['word2vec']['path']))
#     initW = load_embedding_vectors_word2vec(vocabulary,
#                                            cfg['word_embeddings']['word2vec']['path'],
#                                            cfg['word_embeddings']['word2vec']['binary'])
#     print("word2vec file has been loaded")
#   elif embedding_name == 'glove':
#     # load embedding vectors from the glove
#     print("Load glove file {}".format(cfg['word_embeddings']['glove']['path']))
#     initW = load_embedding_vectors_glove(vocabulary,
#                                         cfg['word_embeddings']['glove']['path'],
#                                         embedding_dimension)
#     print("glove file has been loaded\n")
#     print "size of embedding matrix is: "
#     print initW.shape
#   sess.run(cnn.W.assign(initW))


# ==================================================
# RUN LSTM EPOCHS

validation_metrics = {
  "accuracy":
    tf.contrib.learn.metric_spec.MetricSpec(
      metric_fn=tf.contrib.metrics.streaming_accuracy,
      prediction_key=tf.contrib.learn.prediction_key.PredictionKey.
      CLASSES),
  "precision":
    tf.contrib.learn.metric_spec.MetricSpec(
      metric_fn=tf.contrib.metrics.streaming_precision,
      prediction_key=tf.contrib.learn.prediction_key.PredictionKey.
      CLASSES),
  "recall":
    tf.contrib.learn.metric_spec.MetricSpec(
      metric_fn=tf.contrib.metrics.streaming_recall,
      prediction_key=tf.contrib.learn.prediction_key.PredictionKey.
      CLASSES)
}

no_of_batches = int(NUM_EXAMPLES/BATCH_SIZE)
n_epochs = 10

for i in range(n_epochs):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = x_train[ptr:ptr+BATCH_SIZE], y_train[ptr:ptr+BATCH_SIZE]
        ptr+=BATCH_SIZE
        sess.run(minimize ,{data: inp, target: out})

        if j % no_of_batches == 0:
          # Calculate batch accuracy and loss
          acc = sess.run(accuracy, {data: inp, target: out})
          loss = sess.run(cross_entropy, {data: inp, target: out})
          print "Iter " + str(i*BATCH_SIZE) + ", Minibatch Loss= " + \
            "{:.6f}".format(loss) + ", Training Accuracy= " + \
            "{:.5f}".format(acc)

          # Calculate test accuracy and loss
          acc = sess.run(accuracy, {data: x_dev, target: y_dev})
          loss = sess.run(cross_entropy, {data: x_dev, target: y_dev})
          print "Testing Loss= " + "{:.6f}".format(loss) + \
                ", Testing Accuracy= " + "{:.5f}".format(acc)

# Final evaluation of test accuracy and loss
acc = sess.run(accuracy, {data: x_dev, target: y_dev})
loss = sess.run(cross_entropy, {data: x_dev, target: y_dev})
print "Testing Loss= " + "{:.6f}".format(loss) + \
      ", Testing Accuracy= " + "{:.5f}".format(acc)
sess.close()

