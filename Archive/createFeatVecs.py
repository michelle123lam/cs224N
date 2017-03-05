
# Creates feature vectors from Enron Organizational Hierarchy json data
# CS 224N Winter 2017

import json
import argparse
import string
import re
import csv

def get_power_labels_and_indices():
  """Returns power labels for each email"""
  labelled_tuples = {}

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
      for j in range(0, len(recipients)):
        # dominant sender, subordinate recipient = label 0
        if (int(sender), int(recipients[j])) in dominance_map:
          labelled_tuples[(email["uid"], int(sender), int(recipients[j]))] = 0
        elif (int(recipients[j]), int(sender)) in dominance_map:
          labelled_tuples[(email["uid"], int(recipients[j]), int(sender))] = 1

  print len(labelled_tuples)

  return labelled_tuples

get_power_labels_and_indices()

def bag_of_words_features():
  """Returns bag-of-words features for each email"""
  feats = {}
  filename = "enron_database/emails_fixed.json"

  with open(filename, 'r') as f:
    emails = json.load(f) # dictionary of emails

  # TODO: ignore indices that don't have power label

  punctuation = ''.join(string.punctuation)
  whitespace_punc_regex1 = r'([a-zA-Z0-9])([' + punctuation + '])'
  whitespace_punc_regex2 = r'([' + punctuation + '])([a-zA-Z0-9])'

  for email_id, email in emails.iteritems():
    words = []

    # Discard emails without necessary fields
    if "subject" not in email or "body" not in email or "recipients" not in email or len(email["recipients"]) != 1 or "sender" not in email:
      continue

    # Process email subject
    subject = email["subject"]
    subject = subject.replace('\n', '')
    subject = re.sub(whitespace_punc_regex1, r'\1 \2', subject)
    subject = re.sub(whitespace_punc_regex2, r'\1 \2', subject)
    words.extend(subject.split())

    # Process email subject
    body = email["body"]
    body = body.replace('\n', '')
    body = re.sub(whitespace_punc_regex1, r'\1 \2', body)
    body = re.sub(whitespace_punc_regex2, r'\1 \2', body)
    words.extend(body.split())

    recipient = email["recipients"][0] # TODO: handle multiple recipients
    sender = email["sender"]

    # key = (int(email_id), int(sender), int(recipient))
    key = "(" + str(email_id) + ", " + str(sender) + ", " + str(recipient) + ")" # key for json
    feats[key] = words # add email's bag-of-words to feature vector

  # Save to json
  with open('feat_vecs.json', 'w') as fp:
    json.dump(feats, fp)

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

    # TEMP: print the first 5 feature vectors
    i = 0
    for key in feat_vecs:
      if i > 5:
        break
      print key, feat_vecs[key]
      i += 1


if __name__ == "__main__":
    main()
