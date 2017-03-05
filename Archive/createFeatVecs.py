
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
  with open('./enron_database/emails_fixed.json') as file:
    d_emails = json.load(file)
    for i in range(0, 276279):
      email = d_emails[str(i)]

      # check from
      if "from" not in email:
        continue
      sender = email["from"]
      if sender is None:
        continue

      # check to
      if "recipients" not in email:
        continue
      recipients = email["recipients"]
      if recipients is None:
        continue
      if len(recipients) != 1:
        continue

      # dominant sender, subordinate recipient = label 0
      if (int(sender), int(recipients[0])) in dominance_map:
        indices_with_power_relations.append(i)
        labels.append(0)
      elif (int(recipients[0]), int(sender)):
        indices_with_power_relations.append(i)
        labels.append(1)

  print len(indices_with_power_relations)
  print len(labels)

  print indices_with_power_relations[0]
  print labels[0]

  return indices_with_power_relations, labels

get_power_labels_and_indices()

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