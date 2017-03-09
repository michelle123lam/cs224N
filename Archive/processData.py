
# Loads and processes the Enron Organizational Hierarchy json data
# CS 224N Winter 2017

import json
import argparse
import numpy as np
import re

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

  print "Successfully wrote to %s!" % (new_filename)


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

  print "Successfully wrote to %s!" % (new_filename)


def readJson(filename):
  with open(filename) as data_file:
    data = json.load(data_file)
    print data["0"]


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
    return string.strip().lower()


def load_data_and_labels(email_contents_file, labels_file):
    """
    Splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    x_text = np.load(email_contents_file)
    x_text = [clean_str(email.strip()) for email in x_text]  # Split by words and clean with regex
    labels = np.load(labels_file)
    # x_text contains an array of strings for all examples
    # y contains an array of labels for all examples
    return [x_text, labels]

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