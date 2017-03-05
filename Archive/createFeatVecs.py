
# Creates feature vectors from Enron Organizational Hierarchy json data
# CS 224N Winter 2017

import json
import argparse

def get_power_labels():
  """Returns power labels for each email"""
  labels = []
  # TODO: Cat
  return labels

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