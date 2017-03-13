#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from text_cnn import TextCNN
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
from processData import batch_iter, load_data_and_labels, load_embedding_vectors_word2vec, load_embedding_vectors_glove
from utils.treebank import StanfordSentiment
import utils.glove as glove
import yaml

# Parameters
# ==================================================
# Model Hyperparameters
tf.flags.DEFINE_boolean("enable_word_embeddings", True, "Enable/disable the word embedding (default: True)")
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("l2_reg_lambda", 0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# with open("config.yml", 'r') as ymlfile:
#     cfg = yaml.load(ymlfile)

# if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
#     embedding_name = cfg['word_embeddings']['default']
#     embedding_dimension = cfg['word_embeddings'][embedding_name]['dimension']
# else:
#     embedding_dimension = FLAGS.embedding_dim

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
            print "this word %s does not appear in the glove vector initialization" % word
        else:
            indices.append(tokens[word])

    sentVector = np.mean(wordVectors[indices, :], axis=0)
    print sentVector.shape

    assert sentVector.shape == (wordVectors.shape[1],)
    return sentVector

# Load data
emails, labels = load_data_and_labels("email_contents.npy", "labels.npy")
emails = np.array(emails)
print "The number of e-mails is %d" % len(emails)

max_email_length = max([len(email) for email in emails])
print "The max_email_length is %d" % max_email_length

dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

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

zeros = np.zeros((wordVectors.shape[1],))
# Load train set and initialize with glove vectors.
nTrain = len(x_train)
trainFeatures = np.zeros((nTrain, FLAGS.embedding_dim))  # dimVectors should be embedding_dim
trainLabels = y_train
for i in xrange(nTrain):
    words = x_train[i]
    trainFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

# Prepare test set features
nTest = len(x_test)
testFeatures = np.zeros((nTest, FLAGS.embedding_dim))
testLabels = y_test
for i in xrange(nTest):
    words = x_test[i]
    testFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

# Training
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=max_email_length,
            num_classes=2,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", cnn.loss)
    acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    # Write vocabulary
    vocab_processor.save(os.path.join(out_dir, "vocab"))

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
    #     vocabulary = vocab_processor.vocabulary_
    #     initW = None
    #     if embedding_name == 'word2vec':
    #         # load embedding vectors from the word2vec
    #         print("Load word2vec file {}".format(cfg['word_embeddings']['word2vec']['path']))
    #         initW = load_embedding_vectors_word2vec(vocabulary,
    #                                                              cfg['word_embeddings']['word2vec']['path'],
    #                                                              cfg['word_embeddings']['word2vec']['binary'])
    #         print("word2vec file has been loaded")
    #     elif embedding_name == 'glove':
    #         # load embedding vectors from the glove
    #         print("Load glove file {}".format(cfg['word_embeddings']['glove']['path']))
    #         initW = load_embedding_vectors_glove(vocabulary,
    #                                                           cfg['word_embeddings']['glove']['path'],
    #                                                           embedding_dimension)
    #         print("glove file has been loaded\n")
            
    #     sess.run(cnn.W.assign(initW))

    def train_step(x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
          cnn.input_x: x_batch,
          cnn.input_y: y_batch,
          cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
        }
        _, step, summaries, loss, accuracy = sess.run(
            [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)

    def dev_step(x_batch, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
          cnn.input_x: x_batch,
          cnn.input_y: y_batch,
          cnn.dropout_keep_prob: 1.0
        }
        step, summaries, loss, accuracy = sess.run(
            [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        if writer:
            writer.add_summary(summaries, step)

    # Generate batches
    batches = batch_iter(
        list(zip(trainFeatures, trainLabels)), FLAGS.batch_size, FLAGS.num_epochs)
    # Training loop. For each batch...
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)

        # Hold off on testing
        # if current_step % FLAGS.evaluate_every == 0:
        #     print("\nEvaluation @ Epoch = %d:" % batch)
        #     dev_step(testFeatures, testLabels, writer=dev_summary_writer)
        #     print("")
        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))


