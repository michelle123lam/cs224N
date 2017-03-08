
# Creates feature vectors from Enron Organizational Hierarchy json data
# CS 224N Winter 2017

import json
import argparse
import string
import re
import csv
from sklearn.feature_extraction.text import CountVectorizer
from random import shuffle
import numpy as np

def get_power_labels_and_indices(d_emails):
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
  # with open('./enron_database/emails_fixed.json') as file:
  #   d_emails = json.load(file)
  email_contents = [] # text content of each email
  email_keys = [] # key of each email (email uid, sender uid, recipient uid)

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

    # check valid text content
    if "subject" not in email and "body" not in email:
      continue

    for j in range(0, len(recipients)):
      # cur_key = (email["uid"], int(sender), int(recipients[j]))
      cur_key = "(" + str(email["uid"]) + ", " + str(sender) + ", " + str(recipients[j]) + ")" # key for json
      content = ""
      if "subject" in email:
        content += email["subject"]
      if "body" in email:
        content += email["body"]

      content = content.encode('utf-8').replace('\n', '')

      # dominant sender, subordinate recipient = label 0
      if (int(sender), int(recipients[j])) in dominance_map:
        labelled_tuples[cur_key] = 0
        email_contents.append(content) # Add email contents to list
        email_keys.append(cur_key)
      elif (int(recipients[j]), int(sender)) in dominance_map:
        # labelled_tuples[(email["uid"], int(recipients[j]), int(sender))] = 1
        labelled_tuples[cur_key] = 1
        email_contents.append(content) # Add email contents to list
        email_keys.append(cur_key)

  print len(labelled_tuples)

  return labelled_tuples, email_contents, email_keys


def bag_of_words_features(labels, d_emails):
  """Returns bag-of-words features for each email"""
  feats = {}
  feats_and_labels = {}
  # filename = "enron_database/emails_fixed.json"
  # with open(filename, 'r') as f:
  #   emails = json.load(f) # dictionary of emails

  # TODO: ignore indices that don't have power label

  punctuation = ''.join(string.punctuation)
  whitespace_punc_regex1 = r'([a-zA-Z0-9])([' + punctuation + '])'
  whitespace_punc_regex2 = r'([' + punctuation + '])([a-zA-Z0-9])'

  for email_id, email in d_emails.iteritems():
    words = []

    # Discard emails without necessary fields
    if "subject" not in email or "body" not in email or "recipients" not in email or len(email["recipients"]) == 0 or "sender" not in email:
      continue

    recipients = email["recipients"]
    sender = email["sender"]
    for recipient in recipients:
      cur_key = (email["uid"], sender, recipient)
      if cur_key in labels:

        # Process email subject
        subject = email["subject"]
        subject = subject.replace('\n', '')
        subject = re.sub(whitespace_punc_regex1, r'\1 \2', subject)
        subject = re.sub(whitespace_punc_regex2, r'\1 \2', subject)
        words.extend(subject.split())

        # Process email body
        body = email["body"]
        body = body.replace('\n', '')
        body = re.sub(whitespace_punc_regex1, r'\1 \2', body)
        body = re.sub(whitespace_punc_regex2, r'\1 \2', body)
        words.extend(body.split())

        # key = (int(email["uid"]), int(sender), int(recipient))
        key = "(" + str(email["uid"]) + ", " + str(sender) + ", " + str(recipient) + ")" # key for json
        feats[key] = words # add email's bag-of-words to feature vector
        feats_and_labels[key] = [words, labels[cur_key]] # adds email's features and label

  print "Completed feat_vecs!"
  # Save to json
  # with open('feat_vecs.json', 'w') as fp:
  #   json.dump(feats, fp)
  with open('feats_and_labels.json', 'w') as fp:
    json.dump(feats_and_labels, fp)

  print "Completed storing feat_and_labels to json!"
  return feats, feats_and_labels

# def getEmailContents(labels, d_emails):
#   email_contents = []

#   for email_id, email in d_emails.iteritems():
#     # Discard emails without necessary fields
#     if "subject" not in email or "body" not in email or "recipients" not in email or len(email["recipients"]) == 0 or "sender" not in email:
#       continue

#     recipients = email["recipients"]
#     sender = email["sender"]
#     for recipient in recipients:
#       cur_key = (email["uid"], sender, recipient)

#       # Only consider emails with power relations
#       if cur_key in labels:
#         content = email["subject"] + email["body"]
#         email_contents.append(content) # Add email contents to list

  print "Completed storing contents of all emails in list!"

def load_emails():
  emails = {}
  for file_i in range(1, 2):
    # Load json; add its data to emails list
    cur_filename = "enron_database/emails_fixed_%d.json" % (file_i)
    f = open(cur_filename, 'r')
    data = json.load(f)
    emails.update(data)
  return emails

# 94,273 gender "labelled"
# 58,490 are actually labelled with Male or Female categories.
def load_gender_map():
  # Create a map associating UID to gender
  gender_map = {}
  num_nonclassified = 0
  with open("Columbia_Enron_FirstName_Gender_Type.csv") as file:
    d_reader = csv.reader(file, delimiter = ",")
    for row in d_reader:
      uid = row[0]
      gender = row[5]
      if gender == "FALSE":
        break
      if gender == "-1":
        num_nonclassified += 1
      gender_map[uid] = gender
  print "The number of non-classified employees is " + str(num_nonclassified)
  return gender_map

def get_gender_features():
  # Feature vector includes tuples with (gender of sender, gender of recipient)
  # 1 represents Male, 0 represents Female, -1 represents unknown
  emails = load_emails()
  gender_map = load_gender_map()
  print len(gender_map.keys())
  gender_feature_vector = []

  for email_id, email in emails.iteritems():
    sender = str(email["from"])
    recipient = str(email["recipients"][0])  # To-do: Handle multiple recipients

    # Skip the e-mail cases where either the sender or recipient is None, or if there is more than one recipient
    if sender is None or recipient is None:
      continue

    sender_recipient_tuple = (gender_map.get(sender, -1), gender_map.get(recipient, -1))
    print sender_recipient_tuple
    gender_feature_vector.append(sender_recipient_tuple)

  print "length of gender feature vector is " + str(len(gender_feature_vector))
  # length of gender feature vector is 50,000
  return gender_feature_vector

# get_gender_features()

train = 0.7
dev = 0.15
test = 0.15
def generate_split_dataset(input, labels):
  indices = range(train_counts.shape[0])
  shuffle(indices)

  #get shuffled inputs and labels


def process_command_line():
  """Sets command-line flags"""
  parser = argparse.ArgumentParser(description="Write and read formatted Json files")
  # optional arguments
  parser.add_argument('--bow', dest='is_bow', type=bool, default=False, help='Chooses to generate bag-of-words features')
  args = parser.parse_args()
  return args

def main():
  args = process_command_line()

  if args.is_bow:
    with open('./enron_database/emails_fixed.json') as file:
      d_emails = json.load(file)
      labels, email_contents, email_keys = get_power_labels_and_indices(d_emails)
      print "Finished getting power labels and indices!"

      # save email_contents, email_keys, and labels to files
      email_contents = np.array(email_contents)
      email_keys = np.array(email_keys)
      np.save('email_contents.npy', email_contents)
      with open('email_contents.txt','wb') as f:
        np.savetxt(f, email_contents, delimiter='\n', fmt="%s")

      np.save('email_keys.npy', email_keys)
      with open('email_keys.txt','wb') as f:
        np.savetxt(f, email_keys, delimiter='\n', fmt="%s")

      np.save('labels.npy', labels)
      with open('labels.json','w') as f:
        json.dump(labels, f)
      print "Finished saving email_contents, email_keys, and labels to files!"

      # transform email_contents to sparse vectors of word counts
      count_vect = CountVectorizer()
      train_counts = count_vect.fit_transform(email_contents)
      print "all input vectors shape:", train_counts.shape
      np.save('all_input_counts.npy', train_counts)
      np.savetxt('all_input_counts.txt', train_counts)
      print "all_input_counts counts:", train_counts[:5]

      generate_split_dataset(train_counts, labels)
      print labels

    print "Completed labels and feature vectors!"

if __name__ == "__main__":
    main()
