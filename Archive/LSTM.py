"""
RNN using LSTM cells
Michelle Lam - Winter 2017
"""

import numpy as np
import random
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
from processData import batch_iter, load_data_and_labels, load_data_and_labels_bow
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from utils.treebank import StanfordSentiment
import utils.glove as glove
from nltk import tokenize
np.set_printoptions(threshold=np.inf)

# ==================================================
# PARAMETERS

tf.flags.DEFINE_boolean("get_stats", False, "Get stats on input data (default: False)")

# Feature representation settings
tf.flags.DEFINE_boolean("use_grouped", False, "Enable/disable grouped (sender, recipient) features (default: False)")
tf.flags.DEFINE_boolean("use_word_embeddings", False, "Enable/disable the word embedding (default: False)")
tf.flags.DEFINE_boolean("use_sentence_vec", False, "Enable/disable sentence vector representation (default: False)")
tf.flags.DEFINE_boolean("use_non_lexical", False, "Enable/disable additional non-lexical features (default: False)")
tf.flags.DEFINE_integer("num_non_lexical", 2, "Number of additional non-lexical features (default: 2)")


# Model parameters
tf.flags.DEFINE_boolean("use_dropout", False, "Enable/disable dropout (default: False)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("num_hidden", 60, "Number of training epochs (default: 60)")
tf.flags.DEFINE_float("learning_rate", 0.01, "Number of training epochs (default: 0.01)")
tf.flags.DEFINE_integer("max_email_length", 100, "Max number of emails (or max number of *words* per sentence if use_sentence_vec=True)(default: 100)")
# tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 128)")
# tf.flags.DEFINE_integer("l2_reg_lambda", 0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 1000, "Batch Size (default: 100)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 10)")
tf.flags.DEFINE_integer("experiment_num", 0, "ID of current experiment (default: 0)")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr, value))
print("")


# ==================================================
# HELPER FUNCTIONS

def getDataStats(emails, tokens, wordVectors):
  """
  Gets stats on input data for greater insight into parameter choices
  """
  num_word_vals = []

  # Num of words w/ glove representation (mean, stddev, max, min)
  if tokens is not None and wordVectors is not None:
    num_word_vals_glove = [[1 for word in email if tokens.get(word, 0) != 0] for email in emails]
    num_word_vals = [len(num_word_row) for num_word_row in num_word_vals_glove]
  else:
  # Num of words (mean, stddev, max, min)
    num_word_vals = [len(email.split(" ")) for email in emails]

  mean_words = np.mean(num_word_vals)
  stddev_words = np.std(num_word_vals)
  max_words = np.amax(num_word_vals)
  min_words = np.amin(num_word_vals)

  print "mean_words: %d, stddev_words: %d, max_words: %d, min_words: %d" % (mean_words, stddev_words, max_words, min_words)


def plotAccuracyVsTime(num_epochs, train_metrics, test_metrics, filename, y_var_name):
  """
  Plots metric (ex: accuracy, loss) over all epochs for train and test sets
  """
  x_values = [i + 1 for i in range(num_epochs)]
  plt.figure()
  plt.plot(x_values, train_metrics)
  plt.plot(x_values, test_metrics)
  plt.xlabel("epoch")
  plt.ylabel(y_var_name)
  plt.ylim(ymin=0)
  plt.legend(['train', 'test'], loc='upper left')
  dir = './lstm_logs/plots/'
  plt.savefig(dir + filename)


def getWordVectorFeatures(tokens, wordVectors, email, max_email_length, isTest=False):
    """
    Finds word-level features given an email
    Inputs:
    - tokens -- a dictionary that maps words to their indices in
              the word vector list
    - wordVectors -- word vectors (each row) for all tokens
    - email -- a list of words in the email of interest
    Output:
    - result: feature vector for the email; (n_words, word_vec_dims) size
    """
    result = np.zeros((max_email_length, wordVectors.shape[1],))
    emailVector = [wordVectors[tokens[word]] for word in email if tokens.get(word, 0) != 0]
    emailVector = np.array(emailVector)

    if emailVector.shape[0] > max_email_length: # trim to be size of max accepted num words
      emailVector = emailVector[:max_email_length]

    if emailVector.shape[0] != 0: # at least 1 matching word vector
      result[:emailVector.shape[0], :emailVector.shape[1]] = emailVector


    if FLAGS.use_sentence_vec:
      return result

    # Update length of current feature (for word-level features)
    if isTest:
      testFeature_lens.append(emailVector.shape[0])
    else:
      trainFeature_lens.append(emailVector.shape[0])
    return result


def getSentenceVectorFeatures(tokens, wordVectors, email, max_sentence_length, isTest=False):
    """
    Finds sentence vectors by concatenating the word vectors for the first [max_sentence_length] words in each sentence
    Then sums all sentence vectors for each email
    Output:
    - result: feature vector for the email; (n_words, word_vec_dims) size
    """
    result = np.zeros((max_email_length, wordVectors.shape[1],))
    sentences = tokenize.sent_tokenize(email)
    # print "len(sentences):", len(sentences)

    # Gets list of word vectors for each sentence
    sentenceFeatures = [getWordVectorFeatures(tokens, wordVectors, sentences[i], max_sentence_length, isTest=isTest) for i in range(len(sentences))]
    sentenceFeatures = np.array(sentenceFeatures)
    # print "sentenceFeatures.shape", sentenceFeatures.shape

    # Concatenate word vectors for each sentence and sum all sentence vectors
    #(n_sentences, n_words, word_vec_dims)
    sentenceFeatures = np.sum(sentenceFeatures, axis=0)

    # print "result.shape:", result.shape

    # Update length of current feature (for word-level features)
    # TODO: update to reflect variations in sentence length
    if sentenceFeatures.shape == ():
      max_sentence_length = 0

    if isTest:
      testFeature_lens.append(max_sentence_length)
    else:
      trainFeature_lens.append(max_sentence_length)

    if sentenceFeatures.shape == ():
      return result
    return sentenceFeatures


# ==================================================
# LOAD + PREPARE  DATA

# Assign correct data source
email_contents_file = ""
labels_file = ""
if FLAGS.use_grouped:
  # Use features grouped by (sender, recipient) pairs
  email_contents_file = "email_contents_grouped.npy"
  labels_file = "labels_grouped.npy"
else:
  # Use features for each email
  email_contents_file = "email_contents.npy"
  labels_file = "labels.npy"

# Load non-lexical features
if FLAGS.use_non_lexical:
  num_recipients_features = np.array(np.load("num_recipients_features.npy"))

if FLAGS.use_word_embeddings:
  # ==================================================
  # Processing for concatenated word vector features

  emails, labels = load_data_and_labels(email_contents_file, labels_file)
  emails = np.array(emails)

  # Randomly shuffle data
  np.random.seed(10)
  shuffle_indices = np.random.permutation(np.arange(len(labels)))  # Array of random numbers from 1 to # of labels.
  emails_shuffled = emails[shuffle_indices]
  labels_shuffled = labels[shuffle_indices]

  # Split data into train and test set
  train = 0.7
  dev = 0.3
  x_train, x_test, y_train, y_test = train_test_split(emails_shuffled, labels_shuffled, test_size=0.3, random_state=42)
  # x_train = x_train[:1000]
  # x_test = x_test[:1000]
  # y_train = y_train[:1000]
  # y_test = y_test[:1000]
  print "x_train", x_train.shape
  print "x_test", x_test.shape
  print "y_train", y_train.shape
  print "y_test", y_test.shape

  emails = np.array(emails)
  print "emails shape:", emails.shape

  max_email_length = FLAGS.max_email_length
  if FLAGS.use_sentence_vec:
    print "The max words in sentence is %d" % max_email_length
  else:
    print "The max_email_length is %d" % max_email_length

  dataset = StanfordSentiment()
  tokens = dataset.tokens()
  nWords = len(tokens)
  embedding_dim = 100

  # Initialize word vectors with glove.
  wordVectors = glove.loadWordVectors(tokens)
  print "The shape of embedding matrix is:"
  print wordVectors.shape  # Should be number of e-mails, number of embeddings

  if FLAGS.get_stats:
    getDataStats(emails, tokens, wordVectors)

  # Load train set and initialize with glove vectors.
  nTrain = len(x_train)

  trainFeatures = np.zeros((nTrain, embedding_dim + FLAGS.num_non_lexical)) #5 is the number of slots the extra features take up
  toRemove = []
  for i in xrange(nTrain):
    words = x_train[i]

    if FLAGS.use_non_lexical:
      # Calculate num_words feature
      num_words = len(words)
      # Place number of words in buckets
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

    if FLAGS.use_sentence_vec:
      sentenceFeatures = getSentenceVectorFeatures(tokens, wordVectors, words, max_email_length)
    else:
      sentenceFeatures = getWordVectorFeatures(tokens, wordVectors, words, max_email_length)

    if sentenceFeatures is None:
      toRemove.append(i)
    else:
      if FLAGS.use_non_lexical:
        # Add on num_recipients, num_words features
        featureVector = np.hstack((sentenceFeatures, num_recipients_features[i]))
        featureVector = np.hstack((featureVector, num_words_bucket))
        trainFeatures[i, :] = featureVector

  y = np.delete(y, toRemove, axis=0)
  trainFeatures = np.delete(trainFeatures, toRemove, axis=0)





  trainFeature_lens = []
  trainFeatures = []
  if FLAGS.use_sentence_vec:
    trainFeatures = [getSentenceVectorFeatures(tokens, wordVectors, x_train[i], max_email_length) for i in xrange(nTrain)]
  else:
    trainFeatures = [getWordVectorFeatures(tokens, wordVectors, x_train[i], max_email_length) for i in xrange(nTrain)]
  trainFeatures = np.array(trainFeatures)
  trainLabels = y_train
  print "Completed trainFeatures!"
  print "trainFeatures", trainFeatures.shape
  print "trainLabels", trainLabels.shape
  print "trainFeature_lens", trainFeature_lens[:10]

  # Prepare test set features
  nTest = len(x_test)
  testFeature_lens = []
  testFeatures = []
  if FLAGS.use_sentence_vec:
    testFeatures = [getSentenceVectorFeatures(tokens, wordVectors, x_test[i], max_email_length, isTest=True) for i in xrange(nTest)]
  else:
    testFeatures = [getWordVectorFeatures(tokens, wordVectors, x_test[i], max_email_length, isTest=True) for i in xrange(nTest)]
  testFeatures = np.array(testFeatures)
  testLabels = y_test
  print "Completed testFeatures!"
  print "testFeatures", testFeatures.shape
  print "testLabels", testLabels.shape
  print "testFeature_lens", testFeature_lens[:10]

else:
  # ==================================================
  # Processing for bag-of-words (vector of word counts per email)

  x_text, labels = load_data_and_labels_bow(email_contents_file, labels_file)

  # Build vocabulary
  max_email_length = max([len(x.split(" ")) for x in x_text])
  print "The max_email_length is %d" % max_email_length

  if FLAGS.get_stats:
    getDataStats(x_text, None, None)

  # Function that maps each email to sequences of word ids. Shorter emails will be padded.
  vocab_processor = learn.preprocessing.VocabularyProcessor(max_email_length)

  # x is a matrix where each row contains a vector of integers corresponding to a word.
  emails = np.array(list(vocab_processor.fit_transform(x_text)))
  print "emails shape: ", emails.shape
  print "emails: ", emails[:5][:10]

  # Randomly shuffle data
  np.random.seed(10)
  shuffle_indices = np.random.permutation(np.arange(len(labels)))  # Array of random numbers from 1 to # of labels.
  emails_shuffled = emails[shuffle_indices]
  labels_shuffled = labels[shuffle_indices]

  # Split data into train and test set
  train = 0.7
  dev = 0.3
  trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(emails_shuffled, labels_shuffled, test_size=0.3, random_state=42)

  # Account for 1 timestep
  trainFeatures = np.expand_dims(trainFeatures, axis=1)
  testFeatures = np.expand_dims(testFeatures, axis=1)
  print "trainFeatures", trainFeatures.shape
  print "trainLabels", trainLabels.shape
  print "testFeatures", testFeatures.shape
  print "testLabels", testLabels.shape


# ==================================================
# LSTM

NUM_EXAMPLES = trainFeatures.shape[0] # 47411
N_INPUT = trainFeatures.shape[2] # 14080
RNN_HIDDEN = FLAGS.num_hidden
LEARNING_RATE = FLAGS.learning_rate
keep_rate = FLAGS.dropout_keep_prob
N_CLASSES = 2

BATCH_SIZE = FLAGS.batch_size

if FLAGS.use_word_embeddings:
  N_TIMESTEPS = max_email_length
else:
  N_TIMESTEPS = 1


data = tf.placeholder(tf.float32, [None, N_TIMESTEPS, N_INPUT]) # (batch_size, n_timesteps, n_features)
target = tf.placeholder(tf.float32, [None, N_CLASSES]) # (batch_size, n_classes)
if FLAGS.use_word_embeddings:
  X_lengths = tf.placeholder(tf.int32, [None]) # (batch_size); contains number of words in each sentence (to handle padding)

cell = tf.contrib.rnn.BasicLSTMCell(RNN_HIDDEN)
if FLAGS.use_dropout:
  cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=keep_rate, output_keep_prob=keep_rate)
if FLAGS.use_word_embeddings:
  rnn_outputs, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32, sequence_length=X_lengths)
else:
  rnn_outputs, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
print "rnn_outputs1", rnn_outputs.get_shape()
rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])
print "rnn_outputs2", rnn_outputs.get_shape()
rnn_outputs = rnn_outputs[-1]
print "rnn_outputs3", rnn_outputs.get_shape()

weight = tf.Variable(tf.truncated_normal([RNN_HIDDEN, N_CLASSES]))
bias = tf.Variable(tf.constant(0.1, shape=[N_CLASSES]))
prediction = tf.nn.softmax(tf.matmul(rnn_outputs, weight) + bias)
print "weight", weight.get_shape()
print "bias", bias.get_shape()
print "prediction", prediction.get_shape()

with tf.name_scope('cross_entropy'):
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=target))
  optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
  minimize = optimizer.minimize(cross_entropy)
tf.summary.scalar('cross-entropy', cross_entropy)

# Evaluate model
with tf.name_scope('accuracy'):
  y_p = tf.argmax(prediction, 1)
  y_t = tf.argmax(target, 1)
  with tf.name_scope('correct_pred'):
    correct_pred = tf.equal(y_t, y_p)
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# summ_train= tf.summary.merge_all('train')
# summ_test = tf.summary.merge_all('test')
summ_merged = tf.summary.merge_all()

# Execution of the graph
init_op = tf.global_variables_initializer()
sess = tf.Session()
# Generate summary data
writer_train = tf.summary.FileWriter('./lstm_logs/' + str(FLAGS.experiment_num) + '/train', sess.graph)
writer_test = tf.summary.FileWriter('./lstm_logs/' + str(FLAGS.experiment_num) + '/test', sess.graph)
sess.run(init_op)



# ==================================================
# RUN LSTM EPOCHS

no_of_batches = int(NUM_EXAMPLES/BATCH_SIZE) # num batches in one epoch
n_epochs = FLAGS.num_epochs

train_accuracies = []
train_losses = []
test_accuracies = []
test_losses = []

for i in range(n_epochs):
    ptr = 0
    for j in range(no_of_batches):
      inp, out = trainFeatures[ptr:ptr+BATCH_SIZE], trainLabels[ptr:ptr+BATCH_SIZE]
      feed_dict = {
        data: inp,
        target: out,
      }
      if FLAGS.use_word_embeddings:
        x_lens = trainFeature_lens[ptr:ptr+BATCH_SIZE]
        feed_dict[X_lengths] = x_lens
      ptr+=BATCH_SIZE
      s_train, _ = sess.run([summ_merged, minimize], feed_dict)
      writer_train.add_summary(s_train, (i * no_of_batches) + j) # Write to TensorBoard1
      if j % no_of_batches == 0:
        # Calculate batch accuracy and loss
        acc, loss, s_train = sess.run([accuracy, cross_entropy, summ_merged], feed_dict)
        train_accuracies.append(acc)
        train_losses.append(loss)
        writer_train.add_summary(s_train, (i * no_of_batches) + j) # Write to TensorBoard2
        print "Iter " + str(i*no_of_batches) + ", Minibatch Loss= " + \
          "{:.6f}".format(loss) + ", Training Accuracy= " + \
          "{:.5f}".format(acc)

        # Calculate test accuracy and loss
        feed_dict = {
          data: testFeatures,
          target: testLabels,
        }
        if FLAGS.use_word_embeddings:
          feed_dict[X_lengths] = testFeature_lens
        acc, loss, s_test = sess.run([accuracy, cross_entropy, summ_merged], feed_dict)
        test_accuracies.append(acc)
        test_losses.append(loss)
        writer_test.add_summary(s_test, (i * no_of_batches) + j) # Write to TensorBoard
        print "Testing Loss= " + "{:.6f}".format(loss) + \
              ", Testing Accuracy= " + "{:.5f}".format(acc)


# Final evaluation of test accuracy and loss
feed_dict = {
  data: testFeatures,
  target: testLabels,
}
if FLAGS.use_word_embeddings:
  feed_dict[X_lengths] = testFeature_lens
acc, loss, y_pred, y_target, s_test = sess.run([accuracy, cross_entropy, y_p, y_t, summ_merged], feed_dict)
writer_test.add_summary(s_test, (i * no_of_batches) + j) # Write to TensorBoard
print "Testing Loss= " + "{:.6f}".format(loss) + \
      ", Testing Accuracy= " + "{:.5f}".format(acc)

# Plot accuracies, losses over time
cur_type = "bow"
if FLAGS.use_word_embeddings:
  cur_type = "seq"
accuracy_plot_name = "LSTMAccuracyPlot (%s %d).png" % (cur_type, FLAGS.experiment_num)
loss_plot_name = "LSTMLossPlot (%s %d).png" % (cur_type, FLAGS.experiment_num)
plotAccuracyVsTime(n_epochs, train_accuracies, test_accuracies, accuracy_plot_name, "accuracy")
plotAccuracyVsTime(n_epochs, train_losses, test_losses, loss_plot_name, "cross-entropy loss")

# Report precision, recall, F1
print "Precision: {}%".format(100*metrics.precision_score(y_target, y_pred, average="weighted"))
print "Recall: {}%".format(100*metrics.recall_score(y_target, y_pred, average="weighted"))
print "f1_score: {}%".format(100*metrics.f1_score(y_target, y_pred, average="weighted"))

sess.close()

