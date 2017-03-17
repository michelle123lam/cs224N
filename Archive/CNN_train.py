#! /usr/bin/env python

import tensorflow as tf
import argparse
import numpy as np
import os
import time
import datetime
from text_cnn import TextCNN
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
from processData import batch_iter, load_data_and_labels, load_data_and_labels_thread, load_embedding_vectors_word2vec, load_embedding_vectors_glove
from utils.treebank import StanfordSentiment
import utils.glove as glove
import yaml
import nltk
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

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
tf.flags.DEFINE_integer("num_epochs", 30, "Number of training epochs (default: 200)")
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

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
    embedding_name = cfg['word_embeddings']['default']
    embedding_dimension = cfg['word_embeddings'][embedding_name]['dimension']
else:
    embedding_dimension = FLAGS.embedding_dim

def extract_entity_names(t):
  entity_names = []

  if hasattr(t, 'label') and t.label:
      if t.label() == 'NE':
          entity_names.append(' '.join([child[0] for child in t]))
      else:
          for child in t:
              entity_names.extend(extract_entity_names(child))

  return entity_names

def get_entity_names(d_emails):
  chunked_emails = []
  entity_names = []

  for email in d_emails:
    email = nltk.sent_tokenize(email)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in email]
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    chunked_emails.extend(nltk.ne_chunk_sents(tagged_sentences, binary=True))

  for tree in chunked_emails:
    entity_names.extend(extract_entity_names(tree))

  return entity_names

# Email level
def load_emails():
    emails, labels = load_data_and_labels("email_contents.npy", "labels.npy")
    print "finished loading e-mails"
    emails = np.array(emails)

    print "getting entity names"
    entity_names = get_entity_names(emails)
    print "entity names are"
    print entity_names

    print "The number of e-mails is %d" % len(emails)

    for entity_name in entity_names:
        emails = [email.replace(entity_name, "BOB") for email in emails]

    max_email_length = max([len(email.split(" ")) for email in emails])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_email_length)
    emails = np.array(list(vocab_processor.fit_transform(emails)))

    print "The max_email_length is %d" % max_email_length
    print "example email: "
    print emails[1]

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(labels)))  # Array of random numbers from 1 to # of labels.
    emails_shuffled = emails[shuffle_indices]
    labels_shuffled = labels[shuffle_indices]

    train = 0.9
    dev = 0.3
    x_train, x_test, y_train, y_test = train_test_split(emails_shuffled, labels_shuffled, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test, vocab_processor

# Thread level
def load_threads():
    threads, thread_labels = load_data_and_labels_thread("thread_content.npy", "thread_labels.npy")
    threads = np.array(threads)
    print "The number of e-mails is %d" % len(threads)

    max_thread_length = max([len(thread.split(" ")) for thread in threads])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_thread_length)
    threads = np.array(list(vocab_processor.fit_transform(threads)))

    print "The max_thread_length is %d" % max_thread_length
    print "example thread: "
    print threads[1]

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(thread_labels)))  # Array of random numbers from 1 to # of labels.
    threads_shuffled = threads[shuffle_indices]
    thread_labels_shuffled = thread_labels[shuffle_indices]

    train = 0.9
    dev = 0.3
    x_train, x_test, y_train, y_test = train_test_split(threads_shuffled, thread_labels_shuffled, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test, vocab_processor

def train_tensorflow(x_train, x_test, y_train, y_test, vocab_processor):
    # Training
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
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

        if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
            vocabulary = vocab_processor.vocabulary_
            initEmbedding = None
            if embedding_name == 'word2vec':
                # load embedding vectors from the word2vec
                print("Load word2vec file {}".format(cfg['word_embeddings']['word2vec']['path']))
                initEmbedding = load_embedding_vectors_word2vec(vocabulary,
                                                                     cfg['word_embeddings']['word2vec']['path'],
                                                                     cfg['word_embeddings']['word2vec']['binary'])
                print("word2vec file has been loaded")
            elif embedding_name == 'glove':
                # load embedding vectors from the glove
                print("Load glove file {}".format(cfg['word_embeddings']['glove']['path']))
                initEmbedding = load_embedding_vectors_glove(vocabulary,
                                                                  cfg['word_embeddings']['glove']['path'],
                                                                  embedding_dimension)
                print("glove file has been loaded\n")
                
            sess.run(cnn.E.assign(initEmbedding))

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
    train_batches = batch_iter(
        list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
    dev_batches = batch_iter(
        list(zip(x_test, y_test)), FLAGS.batch_size, FLAGS.num_epochs)
    # Training loop. For each batch...
    for train_batch, dev_batch in zip(train_batches, dev_batches):
        x_train_batch, y_train_batch = zip(*train_batch)
        train_step(x_train_batch, y_train_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            x_dev_batch, y_dev_batch = zip(*dev_batch)
            dev_step(x_dev_batch, y_dev_batch, writer=dev_summary_writer)
            print("")
        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))

def main():
  args = process_command_line()
  if args.is_email:
    x_train, x_test, y_train, y_test, vocab_processor = load_emails()
    train_tensorflow(x_train, x_test, y_train, y_test, vocab_processor)
  else:
    x_train, x_test, y_train, y_test, vocab_processor = load_threads()
    train_tensorflow(x_train, x_test, y_train, y_test, vocab_processor)

def process_command_line():
  """Sets command-line flags"""
  parser = argparse.ArgumentParser(description="Write and read formatted Json files")
  # optional arguments
  parser.add_argument('--email', dest='is_email', type=bool, default=False, help='Creates labels on the e-mail lavel')
  parser.add_argument('--thread', dest='is_thread', type=bool, default=False, help='Creates labels on the thread level')
  args = parser.parse_args()
  return args

if __name__ == "__main__":
    main()
