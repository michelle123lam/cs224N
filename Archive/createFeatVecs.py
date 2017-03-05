
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

  emails = {}
  for file_i in range(1, 2):
    # Load json; add its data to emails list
    cur_filename = "enron_database/emails_fixed_%d.json" % (file_i)
    f = open(cur_filename, 'r')
    data = json.load(f)
    emails.update(data)

  # TODO: ignore indices that don't have power label

  for email_id, email in emails.iteritems():
    words = []
    subject = email["subject"]
    body = email["body"]
    print "email_id", email_id
    subject = subject.replace('\n', '')
    body = body.replace('\n', '')
    # TODO: more processing of the string? punctuation?
    words.append(subject.split())
    words.append(body.split())
    feats.append(words) # add email's bag-of-words to feature vector

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
    labels = get_power_labels()
    feat_vecs = bag_of_words_features()
    print "feat_vecs:", feat_vecs[:5]


if __name__ == "__main__":
    main()