
# Loads and processes the Enron Organizational Hierarchy json data
# CS 224N Winter 2017

import json
import argparse
import numpy as np
import re
import string
from pymongo import MongoClient

def writeSplitJson(old_filename, new_filename, n_per_file):
  """Write proper version of json"""
  with open(old_filename, 'r') as f:
    i = 0
    file_i = 0
    g = None
    for x in f:
      if i % n_per_file == 0:
        # Wrap up the current json file
        if g is not None:
          print >> g , "}" # Closing bracket
          g.close()

        # Open and start writing to new json file
        file_i += 1
        cur_new_filename = "%s_%d.json" % (new_filename, file_i)
        g = open(cur_new_filename, 'w')
        print >> g, "{" # Opening bracket

      x = x.rstrip()
      if not x: continue

      # Add entry with index as key; line item as value
      new_x = "\"%d\" : %s" % (i, x)

      # Place comma between subsequent entries
      if (i % n_per_file) > 0:
        new_x = "," + new_x
      print >> g, new_x
      i += 1
    print >> g , "}" # Closing bracket

  print("Successfully wrote to %s!" % (new_filename))


def writeJson(old_filename, new_filename):
  """Write proper version of json"""
  with open(old_filename, 'r') as f:
    with open(new_filename, 'w') as g:
      print >> g, "{" # Opening bracket
      i = 0
      for x in f:
        x = x.rstrip()
        if not x: continue
        # Add entry with index as key; line item as value
        new_x = "\"%d\" : %s" % (i, x)
        # Place comma between subsequent entries
        if i > 0:
          new_x = "," + new_x
        print >> g, new_x
        i += 1
      print >> g , "}" # Closing bracket

  print("Successfully wrote to %s!" % (new_filename))


def readJson(filename):
  with open(filename) as data_file:
    data = json.load(data_file)
    print(data["0"])


def process_command_line():
  """Sets command-line flags"""
  parser = argparse.ArgumentParser(description="Write and read formatted Json files")
  # optional arguments
  parser.add_argument('--write', dest='write_json', type=bool, default=False, help='Writes proper version of json files')
  parser.add_argument('--read', dest='read_json', type=bool, default=False, help='Reads proper version of json files')
  args = parser.parse_args()
  return args

def clean_str(string):
     """
     Tokenization/string cleaning for all datasets except for SST.
     Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
     """
     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
     string = re.sub(r"\'s", " \'s", string)
     string = re.sub(r"\'ve", " \'ve", string)
     string = re.sub(r"n\'t", " n\'t", string)
     string = re.sub(r"\'re", " \'re", string)
     string = re.sub(r"\'d", " \'d", string)
     string = re.sub(r"\'ll", " \'ll", string)
     string = re.sub(r",", " , ", string)
     string = re.sub(r"!", " ! ", string)
     string = re.sub(r"\(", " \( ", string)
     string = re.sub(r"\)", " \) ", string)
     string = re.sub(r"\?", " \? ", string)
     string = re.sub(r"\s{2,}", " ", string)
     return string.strip() #.lower()

def load_data_and_labels_bow(email_contents_file, labels_file):
    """
    Splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    x_text = np.load(email_contents_file)
    x_text = [clean_str(email.strip()) for email in x_text]  # Split by words and clean with regex
    labels = np.array(np.load(labels_file))
    labels = [[1, 0] if a == 0 else [0, 1] for a in labels] # [1, 0] for superior sender; [0, 1] fr superior recipient
    labels = np.array(labels)

    # Finish loading the datasets as arrays, then test it with print statement

    # x_text contains an array of strings for all examples
    # y contains an array of labels for all examples
    return [x_text, labels]

def load_data_and_labels_thread(thread_content_file, thread_labels, non_lexical_features):
  thread_content = np.load(thread_content_file)
  thread_content = [clean_str(emails.strip()) for emails in thread_content]  # Split by words and clean with regex
  thread_labels = np.array(np.load(thread_labels))
  thread_labels = [[1, 0] if a == 0 else [0, 1] for a in thread_labels] # [1, 0] for superior sender; [0, 1] fr superior recipient
  thread_labels = np.array(thread_labels)
  non_lexical_features = np.load(non_lexical_features)

  return [thread_content, thread_labels, non_lexical_features]

def load_data_and_labels(email_contents_file, labels_file):
    """
    Splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    email_contents = np.load(email_contents_file)
    email_contents = [clean_str(email) for email in email_contents]
    email_contents = np.array(email_contents)

    # Number of emails is: 67,730
    labels = np.array(np.load(labels_file))
    labels = [[1, 0] if a == 0 else [0, 1] for a in labels] # [1, 0] for superior sender; [0, 1] for superior recipient
    labels = np.array(labels)

    return [email_contents, labels]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
  """
  Generates a batch iterator for a dataset.
  """
  data = np.array(data)
  data_size = len(data)
  num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
  for epoch in range(num_epochs):
      # Shuffle the data at each epoch
      if shuffle:
          shuffle_indices = np.random.permutation(np.arange(data_size))
          shuffled_data = data[shuffle_indices]
      else:
          shuffled_data = data
      for batch_num in range(num_batches_per_epoch):
          start_index = batch_num * batch_size
          end_index = min((batch_num + 1) * batch_size, data_size)
          yield shuffled_data[start_index:end_index]

def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    # load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word), encoding=encoding, errors='strict')
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, vector = parts[0], list(map('float32', parts[1:]))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors


def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors

def main():
  args = process_command_line()

  if args.write_json:
    n_per_file = 50000 # Number of items per json file
    writeJson('enron_database/emails.json', 'enron_database/emails_fixed.json')
    writeJson('enron_database/entities.json', 'enron_database/entities_fixed.json')
    writeJson('enron_database/threads.json', 'enron_database/threads_fixed.json')
  if args.read_json:
    readJson('enron_database/emails_fixed.json')
    readJson('enron_database/entities_fixed.json')
    readJson('enron_database/threads_fixed.json')

if __name__ == "__main__":
    main()