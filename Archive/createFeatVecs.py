
# Creates feature vectors from Enron Organizational Hierarchy json data
# CS 224N Winter 2017

import json
import argparse
import csv

def get_power_labels_and_indices():
  """Returns power labels for each email"""
  indices_with_power_relations = []
  labels = []

  dominance_map = {}
  # read in dominance tuples file in the form (boss, subordinate): immediate?
  with open('Columbia_Enron_DominanceTuples.csv') as file:
    d_reader = csv.reader(file, delimiter=",")
    rows_read = 0
    for row in d_reader:
      if rows_read != 0:
        dominance_map[(int(row[0]), int(row[1]))] = int(row[2])
      rows_read += 1

  # read in email tuples and check if power relations exist for the email
  with open('emails_fixed.json'):
  return indices_with_power_relations, labels

get_power_labels_and_indices()

def bag_of_words_features():
  """Returns bag-of-words features for each email"""
  feats = []
  # TODO: Michelle
  return feats

def process_command_line():
  """Sets command-line flags"""
  parser = argparse.ArgumentParser(description="Write and read formatted Json files")
  # optional arguments
  parser.add_argument('--bow', dest='is_bow', type=bool, default=False, help='Chooses to generate bag-of-words features')
  args = parser.parse_args()
  return args


def main():
  args = process_command_line()

  feat_vecs = []
  if args.is_bow:
    feat_vecs = bag_of_words_features()
    labels = get_power_labels()


if __name__ == "__main__":
    main()