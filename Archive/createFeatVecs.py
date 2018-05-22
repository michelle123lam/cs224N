# Creates feature vectors from Enron Organizational Hierarchy json data
# CS 224N Winter 2017

import nltk
import json
import argparse
import string
import re
import math
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from random import shuffle
import numpy as np
from random import randint
from processData import clean_str
from pymongo import MongoClient

# Constants and global variables
CONST_NUM_EMAILS = 276279
CONST_NUM_THREADS = 36196
emails_map = {}  # Map from email u_id to the email content
all_email_names_set = set()

stopList = ["i",
"a",
"about",
"an",
"and",
"are",
"as",
"at",
"be",
"by",
"can",
"com",
"day",
"do",
"email",
"for",
"from",
"go",
"hi",
"how",
"I",
"in",
"is",
"it",
"love",
"now",
"me",
"of",
"off",
"on",
"online",
"or",
"one",
"out",
"Please",
"so",
"that",
"the",
"this",
"This",
"to",
"want",
"was",
"way",
"we",
"We",
"what",
"when",
"where",
"who",
"will",
"with",
"you",
"You",
"your",
"the",
"The",
"via",
"www"
"|"]

def populate_email_names_set():
    client = MongoClient()
    db = client.enron
    all_results = db.entities.find({}, {"email_names": 1})
    for result in all_results:
        if "email_names" not in result:
            continue
        email_names = result["email_names"]
        for email_name in email_names:
            if email_name is None:
                continue
            else:
                splits = email_name.split()
                for split in splits:
                    if split in stopList:
                        continue
                    all_email_names_set.add(split)

def replace_names(string):
    new_string = []
    for word in string:
        if word in all_email_names_set:
            new_string.append("<NAME>")
        else:
            new_string.append(word)
    return new_string

def get_power_labels_and_indices_thread(d_threads, d_emails):
  """Return power labels for each thread; each thread will be split by the # of interacting partipant pairs"""
  emails_map = {}  # Assume there is a map from email u_id to the email content
  labels_map = {}

  for i in range(0, CONST_NUM_EMAILS):
    email = d_emails[str(i)]
    emails_map[email["uid"]] = email
  print "Finished creating map for all 276279 emails!!"

  # Read in dominance tuples file in the form of (boss, subordinate): immediate?
  with open('Columbia_Enron_DominanceTuples.csv') as file:
    d_reader = csv.reader(file, delimiter=",")
    rows_read = 0
    for row in d_reader:
      if rows_read != 0:
        cur_key1 = (int(row[0]), int(row[1])) # (dom_id, sub_id)
        cur_key2 = (int(row[1]), int(row[0])) # (sub_id, dom_id)
        if cur_key1 not in labels_map:
          labels_map[cur_key1] = 0
        if cur_key2 not in labels_map:
          labels_map[cur_key2] = 1
      rows_read += 1

  # Generate ordered lists for labels and email_contents
  labels = []
  thread_contents = []
  # Go through each thread to update features
  for i in range(0, CONST_NUM_THREADS):
    thread_contents_map = {}  # (sender_id, recipient_id)
    thread = d_threads[str(i)]
    for thread_node in thread["thread_nodes"]:
      message_id = thread_node.get("message_id", None)
      if message_id is None:
        print "The thread node %s does not have a message id" % thread_node["uid"]
        continue
      print "Message id of thread node is %d" % message_id

      email = emails_map.get(message_id, None)  # Not sure if message id is the same as uid of the email?

      # check valid email
      if email is None:
        print "message id does not exist"
        continue

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

      # Prepare content of the email
      content = ""
      if "subject" in email:
        content += email["subject"]
      if "body" in email:
        content += " " + email["body"]
      content = content.encode('utf-8').replace('\n', '')

      # Generate features for each recipient of email
      for j in range(0, len(recipients)):
        cur_key = (int(sender), int(recipients[j]))

        # Check if we know the correct power relation for this (sender, recipient) pair
        if cur_key in labels_map:
          if cur_key in thread_contents_map:
            # Append to existing email contents for this (sender, recipient) pair
            thread_contents_map[cur_key] = thread_contents_map[cur_key] + content
          else:
            # Create new entry for this (sender, recipient) pair
            thread_contents_map[cur_key] = content

    for cur_key in thread_contents_map: # for each pair found in emails
      labels.append(labels_map[cur_key])
      thread_contents.append(thread_contents_map[cur_key])

  print "# labels: %d, # thread_contents: %d" % (len(labels), len(thread_contents))
  return labels, thread_contents

def get_power_labels_and_indices_thread_1(d_threads, d_emails, is_thread_3):
  """Return power labels for each thread; each thread will be split by the # of interacting partipant pairs"""
  emails_map = {}  # Assume there is a map from email u_id to the email content
  labels_map = {}

  for i in range(0, CONST_NUM_EMAILS):
    email = d_emails[str(i)]
    emails_map[email["uid"]] = email
  print "Finished creating map for all 276279 emails!!"

  # Read in dominance tuples file in the form of (boss, subordinate): immediate?
  with open('Columbia_Enron_DominanceTuples.csv') as file:
    d_reader = csv.reader(file, delimiter=",")
    rows_read = 0
    for row in d_reader:
      if rows_read != 0:
        cur_key1 = (int(row[0]), int(row[1])) # (dom_id, sub_id)
        cur_key2 = (int(row[1]), int(row[0])) # (sub_id, dom_id)
        if cur_key1 not in labels_map:
          labels_map[cur_key1] = 0
        if cur_key2 not in labels_map:
          labels_map[cur_key2] = 1
      rows_read += 1

  # Generate ordered lists for labels and email_contents
  labels = []
  thread_contents_1 = []
  thread_contents_2 = []
  # Go through each thread to update features
  for i in range(0, CONST_NUM_THREADS):
    thread_contents_map = {}  # (sender_id, recipient_id)
    thread = d_threads[str(i)]
    for thread_node in thread["thread_nodes"]:
      message_id = thread_node.get("message_id", None)
      if message_id is None:
        print "The thread node %s does not have a message id" % thread_node["uid"]
        continue
      print "Message id of thread node is %d" % message_id

      email = emails_map.get(message_id, None)  # Not sure if message id is the same as uid of the email?

      # check valid email
      if email is None:
        print "message id does not exist"
        continue

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

      # Prepare content of the email
      content = ""
      if "subject" in email:
        content += email["subject"]
      if "body" in email:
        content += " " + email["body"]
      content = content.encode('utf-8').replace('\n', '')

      # Generate features for each recipient of email
      for j in range(0, len(recipients)):
        cur_key = (int(sender), int(recipients[j]))

        # Check if we know the correct power relation for this (sender, recipient) pair
        if cur_key in labels_map:
          if cur_key in thread_contents_map:
            # Append to existing email contents for this (sender, recipient) pair
            thread_contents_map[cur_key] = thread_contents_map[cur_key] + content
          else:
            # Create new entry for this (sender, recipient) pair
            thread_contents_map[cur_key] = content

    for cur_key in thread_contents_map: # for each pair found in emails
      switch_key = (cur_key[1], cur_key[0])
      if is_thread_3: 
        labels.append(labels_map[cur_key])
        thread_contents_1.append(thread_contents_map[cur_key])
        if switch_key in thread_contents_map:
          thread_contents_2.append(thread_contents_map[switch_key])
        else:
          thread_contents_2.append("NO_EMAILS")
      else: 
        if switch_key in thread_contents_map:
          labels.append(labels_map[cur_key])
          thread_contents_1.append(thread_contents_map[cur_key])
          thread_contents_2.append(thread_contents_map[switch_key])

  print "# labels: %d, # thread_contents_1: %d, # thread_contents_2: %d" % (len(labels), len(thread_contents_1), len(thread_contents_2))
  return labels, thread_contents_1, thread_contents_2

def get_power_labels_and_indices_thread_2(d_threads, d_emails):
  """Return power labels for each thread; each thread will be split by the # of interacting partipant pairs"""
  emails_map = {}  # Assume there is a map from email u_id to the email content
  labels_map = {}

  for i in range(0, CONST_NUM_EMAILS):
    email = d_emails[str(i)]
    emails_map[email["uid"]] = email
  print "Finished creating map for all 276279 emails!!"

  # Read in dominance tuples file in the form of (boss, subordinate): immediate?
  with open('Columbia_Enron_DominanceTuples.csv') as file:
    d_reader = csv.reader(file, delimiter=",")
    rows_read = 0
    for row in d_reader:
      if rows_read != 0:
        cur_key1 = (int(row[0]), int(row[1])) # (dom_id, sub_id)
        cur_key2 = (int(row[1]), int(row[0])) # (sub_id, dom_id)
        if cur_key1 not in labels_map:
          labels_map[cur_key1] = 0
        if cur_key2 not in labels_map:
          labels_map[cur_key2] = 1
      rows_read += 1

  # Generate ordered lists for labels and email_contents
  thread_contents_1 = []
  thread_contents_2 = []
  labels = []
  # Go through each thread to update features
  for i in range(0, CONST_NUM_THREADS):
    thread_contents_map = {}  # (sender_id, recipient_id)
    thread = d_threads[str(i)]
    for thread_node in thread["thread_nodes"]:
      message_id = thread_node.get("message_id", None)
      if message_id is None:
        print "The thread node %s does not have a message id" % thread_node["uid"]
        continue
      print "Message id of thread node is %d" % message_id

      email = emails_map.get(message_id, None)  # Not sure if message id is the same as uid of the email?

      # check valid email
      if email is None:
        print "message id does not exist"
        continue

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

      # Prepare content of the email
      content = ""
      if "subject" in email:
        content += email["subject"]
      if "body" in email:
        content += " " + email["body"]
      content = content.encode('utf-8').replace('\n', '')

      # Generate features for each recipient of email
      for j in range(0, len(recipients)):
        cur_key = (int(sender), int(recipients[j]))

        # Check if we know the correct power relation for this (sender, recipient) pair
        if cur_key in labels_map:
          if cur_key in thread_contents_map:
            # Append to existing email contents for this (sender, recipient) pair
            thread_contents_map[cur_key] = thread_contents_map[cur_key] + [content]
          else:
            # Create new entry for this (sender, recipient) pair
            thread_contents_map[cur_key] = [content]

    for cur_key in thread_contents_map: # for each pair found in emails
      switch_key = (cur_key[1], cur_key[0])
      if switch_key in thread_contents_map:
        for email in thread_contents_map[cur_key]:
          thread_contents_1.append(email)
        thread_contents_1.append('---')
        for switch_email in thread_contents_map[switch_key]:
          thread_contents_2.append(switch_email)
        thread_contents_2.append('---')

  print "# thread_contents_1: %d, # thread_contents_2: %d" % (len(thread_contents_1), len(thread_contents_2))
  return thread_contents_1, thread_contents_2

def get_power_labels_and_indices_thread_4(d_threads, d_emails):
  """Return power labels for each thread; each thread will be split by the # of interacting partipant pairs"""
  emails_map = {}  # Assume there is a map from email u_id to the email content
  labels_map = {}

  for i in range(0, CONST_NUM_EMAILS):
    email = d_emails[str(i)]
    emails_map[email["uid"]] = email
  print "Finished creating map for all 276279 emails!!"

  # Read in dominance tuples file in the form of (boss, subordinate): immediate?
  with open('Columbia_Enron_DominanceTuples.csv') as file:
    d_reader = csv.reader(file, delimiter=",")
    rows_read = 0
    for row in d_reader:
      if rows_read != 0:
        cur_key1 = (int(row[0]), int(row[1])) # (dom_id, sub_id)
        cur_key2 = (int(row[1]), int(row[0])) # (sub_id, dom_id)
        if cur_key1 not in labels_map:
          labels_map[cur_key1] = 0
        if cur_key2 not in labels_map:
          labels_map[cur_key2] = 1
      rows_read += 1

  # Generate ordered lists for labels and email_contents
  thread_contents_1 = []
  thread_contents_2 = []
  labels = []
  # Go through each thread to update features
  for i in range(0, CONST_NUM_THREADS):
    thread_contents_map = {}  # (sender_id, recipient_id)
    thread = d_threads[str(i)]
    for thread_node in thread["thread_nodes"]:
      message_id = thread_node.get("message_id", None)
      if message_id is None:
        print "The thread node %s does not have a message id" % thread_node["uid"]
        continue
      print "Message id of thread node is %d" % message_id

      email = emails_map.get(message_id, None)  # Not sure if message id is the same as uid of the email?

      # check valid email
      if email is None:
        print "message id does not exist"
        continue

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

      # Prepare content of the email
      content = ""
      if "subject" in email:
        content += email["subject"]
      if "body" in email:
        content += " " + email["body"]
      content = content.encode('utf-8').replace('\n', '')

      # Generate features for each recipient of email
      for j in range(0, len(recipients)):
        cur_key = (int(sender), int(recipients[j]))

        # Check if we know the correct power relation for this (sender, recipient) pair
        if cur_key in labels_map:
          if cur_key in thread_contents_map:
            # Append to existing email contents for this (sender, recipient) pair
            thread_contents_map[cur_key] = thread_contents_map[cur_key] + [content]
          else:
            # Create new entry for this (sender, recipient) pair
            thread_contents_map[cur_key] = [content]

    for cur_key in thread_contents_map: # for each pair found in emails
      labels.append(labels_map[cur_key])
      switch_key = (cur_key[1], cur_key[0])
      for email in thread_contents_map[cur_key]:
        thread_contents_1.append(email)
      thread_contents_1.append('---')
      if switch_key in thread_contents_map:
        for switch_email in thread_contents_map[switch_key]:
          thread_contents_2.append(switch_email)
      else:
        thread_contents_2.append('NO_EMAILS')
      thread_contents_2.append('---')

  print "#labels: %d, # thread_contents_1: %d, # thread_contents_2: %d" % (len(labels), len(thread_contents_1), len(thread_contents_2))
  return labels, thread_contents_1, thread_contents_2

def get_power_labels_and_indices_grouped(d_emails):
  """Returns power labels for each (sender, recipient) pair"""
  labels_map = {} # (sender_id, recipient_id): label (0 = dom sender, sub recip), (1 = sub sender, dom recip)
  email_contents_map = {} # (sender_id, recipient_id): all email text shared between these parties

  # read in dominance tuples file in the form (boss, subordinate): immediate?
  with open('Columbia_Enron_DominanceTuples.csv') as file:
    d_reader = csv.reader(file, delimiter=",")
    rows_read = 0
    for row in d_reader:
      if rows_read != 0:
        cur_key1 = (int(row[0]), int(row[1])) # (dom_id, sub_id)
        cur_key2 = (int(row[1]), int(row[0])) # (sub_id, dom_id)
        if cur_key1 not in labels_map:
          labels_map[cur_key1] = 0
        if cur_key2 not in labels_map:
          labels_map[cur_key2] = 1
      rows_read += 1

  # Go through each email to create/update features
  for i in range(0, CONST_NUM_EMAILS):
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

    # Prepare content of the email
    content = ""
    if "subject" in email:
      content += email["subject"]
    if "body" in email:
      content += " " + email["body"]
    content = content.encode('utf-8').replace('\n', '')

    # Generate features for each recipient of email
    for j in range(0, len(recipients)):
      cur_key = (int(sender), int(recipients[j]))

      # Check if we know the correct power relation for this (sender, recipient) pair
      if cur_key in labels_map:
        if cur_key in email_contents_map:
          # Append to existing email contents for this (sender, recipient) pair
          email_contents_map[cur_key] = (email_contents_map[cur_key][0] + content, email_contents_map[cur_key][1] + len(recipients), \
                                         email_contents_map[cur_key][2] + len(clean_str(content.strip())), email_contents_map[cur_key][3] + 1)
        else:
          # Create new entry for this (sender, recipient) pair
          email_contents_map[cur_key] = (content, len(recipients), len(content.split(" ")), 1)

  # Generate ordered lists for labels and email_contents
  labels = []
  email_contents =[]
  avgNumRecipients = []
  avgNumTokensPerEmail = []
  avgNumEmailsPerGroup = 0
  for cur_key in email_contents_map: # for each pair found in emails
    labels.append(labels_map[cur_key])
    email_contents.append(email_contents_map[cur_key][0])
    avgNumRecipients.append(float(email_contents_map[cur_key][1]) / float(email_contents_map[cur_key][3]))
    avgNumTokensPerEmail.append(float(email_contents_map[cur_key][2]) / float(email_contents_map[cur_key][3]))
    avgNumEmailsPerGroup += float(email_contents_map[cur_key][3])

  print "# labels: %d, # email_contents: %d" % (len(labels), len(email_contents))
  print len(avgNumRecipients)
  print len(avgNumTokensPerEmail)
  print "Avg Num Emails Per Group: " + str(avgNumEmailsPerGroup / float(len(email_contents_map)))
  return labels, email_contents, avgNumRecipients, avgNumTokensPerEmail

def get_power_labels_and_indices_grouped_1(d_emails, is_3):
  """Returns power labels for each (sender, recipient) pair"""
  labels_map = {} # (sender_id, recipient_id): label (0 = dom sender, sub recip), (1 = sub sender, dom recip)
  email_contents_map = {} # (sender_id, recipient_id): all email text shared between these parties
  # read in dominance tuples file in the form (boss, subordinate): immediate?
  with open('Columbia_Enron_DominanceTuples.csv') as file:
    d_reader = csv.reader(file, delimiter=",")
    rows_read = 0
    for row in d_reader:
      if rows_read != 0:
        cur_key1 = (int(row[0]), int(row[1])) # (dom_id, sub_id)
        cur_key2 = (int(row[1]), int(row[0])) # (sub_id, dom_id)
        if cur_key1 not in labels_map:
          labels_map[cur_key1] = 0
        if cur_key2 not in labels_map:
          labels_map[cur_key2] = 1
      rows_read += 1

  # Go through each email to create/update features
  for i in range(0, CONST_NUM_EMAILS):
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

    # Prepare content of the email
    content = ""
    if "subject" in email:
      content += email["subject"]
    if "body" in email:
      content += " " + email["body"]
    content = content.encode('utf-8').replace('\n', '')

    # keep track of (sender, recipient) pair and make

    # Generate features for each recipient of email
    for j in range(0, len(recipients)):
      cur_key = (int(sender), int(recipients[j]))

      # Check if we know the correct power relation for this (sender, recipient) pair
      if cur_key in labels_map:
        if cur_key in email_contents_map:
          # Append to existing email contents for this (sender, recipient) pair
          email_contents_map[cur_key] = (email_contents_map[cur_key][0] + content, email_contents_map[cur_key][1] + len(recipients), \
                                         email_contents_map[cur_key][2] + len(clean_str(content.strip())), email_contents_map[cur_key][3] + 1)
        else:
          # Create new entry for this (sender, recipient) pair
          email_contents_map[cur_key] = (content, len(recipients), len(content.split(" ")), 1)

  # Generate ordered lists for labels and email_contents
  labels = []
  email_contents_1 = []
  email_contents_2 = []
  avgNumRecipients_1 = []
  avgNumRecipients_2 = []
  avgNumTokensPerEmail_1 = []
  avgNumTokensPerEmail_2 = []
  avgNumEmailsPerGroup_1 = 0
  avgNumEmailsPerGroup_2 = 0
  usedKeys = {}
  for cur_key in email_contents_map: # for each pair found in emails
    switch_key = (cur_key[1], cur_key[0])
    if not is_3:
      if cur_key not in usedKeys and switch_key not in usedKeys and switch_key in email_contents_map:
        labels.append(labels_map[cur_key])

        email_contents_1.append(email_contents_map[cur_key][0])
        email_contents_2.append(email_contents_map[switch_key][0]) #get (recipient, sender) emails
        avgNumRecipients_1.append(float(email_contents_map[cur_key][1]) / float(email_contents_map[cur_key][3]))
        avgNumRecipients_2.append(float(email_contents_map[switch_key][1]) / float(email_contents_map[switch_key][3]))
        avgNumTokensPerEmail_1.append(float(email_contents_map[cur_key][2]) / float(email_contents_map[cur_key][3]))
        avgNumTokensPerEmail_2.append(float(email_contents_map[switch_key][2]) / float(email_contents_map[switch_key][3]))
        avgNumEmailsPerGroup_1 += float(email_contents_map[cur_key][3])
        avgNumEmailsPerGroup_2 += float(email_contents_map[switch_key][3])
      usedKeys[cur_key] = True
      usedKeys[switch_key] = True
    if is_3:
      if cur_key not in usedKeys and switch_key not in usedKeys:
        labels.append(labels_map[cur_key])
        email_contents_1.append(email_contents_map[cur_key][0])
        avgNumRecipients_1.append(float(email_contents_map[cur_key][1]) / float(email_contents_map[cur_key][3]))
        avgNumTokensPerEmail_1.append(float(email_contents_map[cur_key][2]) / float(email_contents_map[cur_key][3]))
        avgNumEmailsPerGroup_1 += float(email_contents_map[cur_key][3])
        if switch_key in email_contents_map:
          email_contents_2.append(email_contents_map[switch_key][0])
          avgNumRecipients_2.append(float(email_contents_map[switch_key][1]) / float(email_contents_map[switch_key][3]))
          avgNumTokensPerEmail_2.append(float(email_contents_map[switch_key][2]) / float(email_contents_map[switch_key][3]))
          avgNumEmailsPerGroup_2 += float(email_contents_map[switch_key][3])
        else:
          email_contents_2.append("NO_EMAILS")
          avgNumRecipients_2.append(0)
          avgNumTokensPerEmail_2.append(0)

      usedKeys[cur_key] = True
      usedKeys[switch_key] = True

  print "# labels: %d, # email_contents: %d" % (len(labels), len(email_contents_1))
  print len(avgNumRecipients_1)
  print len(avgNumTokensPerEmail_1)
  print "Avg Num Emails Per Group 1: " + str(avgNumEmailsPerGroup_1 / float(len(email_contents_1)))
  print "Avg Num Emails Per Group 2: " + str(avgNumEmailsPerGroup_2 / float(len(email_contents_2)))
  return labels, email_contents_1, email_contents_2, avgNumRecipients_1, avgNumRecipients_2, avgNumTokensPerEmail_1, avgNumTokensPerEmail_2, avgNumEmailsPerGroup_1, avgNumEmailsPerGroup_2

def get_power_labels_and_indices_grouped_2(d_emails, is_grouped_4):
  labels_map = {} # (sender_id, recipient_id): label (0 = dom sender, sub recip), (1 = sub sender, dom recip)
  email_contents_map = {} # (sender_id, recipient_id): all email text shared between a sender and recipient pair

  # read in dominance tuples file in the form (boss, subordinate): immediate?
  with open('Columbia_Enron_DominanceTuples.csv') as file:
    d_reader = csv.reader(file, delimiter=",")
    rows_read = 0
    for row in d_reader:
      if rows_read != 0:
        cur_key1 = (int(row[0]), int(row[1])) # (dom_id, sub_id)
        cur_key2 = (int(row[1]), int(row[0])) # (sub_id, dom_id)
        if cur_key1 not in labels_map:
          labels_map[cur_key1] = 0
        if cur_key2 not in labels_map:
          labels_map[cur_key2] = 1
      rows_read += 1

  # Go through each email to create/update features
  for i in range(0, CONST_NUM_EMAILS):
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

    # Prepare content of the email
    content = ""
    if "subject" in email:
      content += email["subject"]
    if "body" in email:
      content += " " + email["body"]
    content = content.encode('utf-8').replace('\n', '')

    # keep track of (sender, recipient) pair and make

    # Generate features for each recipient of email
    for j in range(0, len(recipients)):
      cur_key = (int(sender), int(recipients[j]))
      # Check if we know the correct power relation for this (sender, recipient) pair
      if cur_key in labels_map:
        if cur_key in email_contents_map:
          # Append to existing email contents for this (sender, recipient) pair
          email_contents_map[cur_key] = (email_contents_map[cur_key][0] + [content], email_contents_map[cur_key][1] + len(recipients), \
                                         email_contents_map[cur_key][2] + len(clean_str(content.strip())), email_contents_map[cur_key][3] + 1)
        else:
          # Create new entry for this (sender, recipient) pair
          email_contents_map[cur_key] = ([content], len(recipients), len(content.split(" ")), 1)

  # Generate ordered lists for labels and email_contents
  email_contents_1 = []
  email_contents_2 = []
  labels = []
  usedKeys = {}
  for cur_key in email_contents_map: # for each pair found in emails
    switch_key = (cur_key[1], cur_key[0])
    if not is_grouped_4:
      if cur_key not in usedKeys and switch_key not in usedKeys and switch_key in email_contents_map:
        labels.append(labels_map[cur_key])
        for email in email_contents_map[cur_key][0]:
          email_contents_1.append(email)
        email_contents_1.append('---')
        for switch_email in email_contents_map[switch_key][0]:
          email_contents_2.append(switch_email)
        email_contents_2.append('---')
    else: 
      if cur_key not in usedKeys and switch_key not in usedKeys:
        labels.append(labels_map[cur_key])
        for email in email_contents_map[cur_key][0]:
          email_contents_1.append(email)
        email_contents_1.append('---')
        if switch_key in email_contents_map:
          for switch_email in email_contents_map[switch_key][0]:
            email_contents_2.append(switch_email)
        else: 
          email_contents_2.append('NO_EMAILS')
        email_contents_2.append('---')
    usedKeys[cur_key] = True
    usedKeys[switch_key] = True

  print "# labels: %d, # email_contents_1: %d, # email_contents_2: %d" % (len(labels), len(email_contents_1), len(email_contents_2))
  return email_contents_1, email_contents_2

def get_power_labels_and_indices(d_emails):
  """Returns power labels for each email"""

  gender_map = load_select_feature_maps()

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

  email_contents = [] # text content of each email

  gender_feature_vector = []

  num_recipients_vector = []

  dom_sub = 0
  sub_dom = 0
  for i in range(0, CONST_NUM_EMAILS):
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
    potential = []
    for j in range(0, len(recipients)):
      cur_key = "(" + str(email["uid"]) + ", " + str(sender) + ", " + str(recipients[j]) + ")" # key for json
      content = ""
      if "subject" in email:
        content += email["subject"]
      if "body" in email:
        content +=  " " + email["body"]

      content = content.encode('utf-8').replace('\n', '')
      content = " ".join(replace_names(content.split()))

      # dominant sender, subordinate recipient = label 0
      if (int(sender), int(recipients[j])) in dominance_map:
        potential.append((content, 0, len(recipients), [gender_map.get(str(sender), 0), gender_map.get(str(recipients[j]), 0)]))
      elif (int(recipients[j]), int(sender)) in dominance_map:
        potential.append((content, 1, len(recipients), [gender_map.get(str(sender), 0), gender_map.get(str(recipients[j]), 0)]))
    if len(potential) > 1:
      index = randint(0, len(potential) - 1)
      email_contents.append(potential[index][0])
      labels.append(potential[index][1])
      if potential[index][1] == 0:
          dom_sub += 1
      else:
          sub_dom += 1
      num_recipients_vector.append(potential[index][2])
      gender_feature_vector.append(potential[index][3])
    elif len(potential) == 1:
      email_contents.append(potential[0][0])
      labels.append(potential[0][1])
      if potential[0][1] == 0:
          dom_sub += 1
      else:
          sub_dom += 1
      num_recipients_vector.append(potential[0][2])
      gender_feature_vector.append(potential[0][3])

  print("Dominant-Subordinates: " + str(dom_sub))
  print("Subordinate-Dominants: " + str(sub_dom))
  print("Total labels: " + str(len(labels)))
  print("Total emails: " + str(len(email_contents)))

  return labels, email_contents, num_recipients_vector, gender_feature_vector

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

# 94,273 gender "labelled"
# 58,490 are actually labelled with Male or Female categories.
def load_select_feature_maps():
  # Create a map associating UID to gender
  gender_map = {}
  employee_type_map = {}
  num_nonclassified = 0

  with open("Columbia_Enron_FirstName_Gender_Type.csv") as file:
    d_reader = csv.reader(file, delimiter = ",")
    rows_read = 0
    for row in d_reader:
      if rows_read != 0:
        uid = row[0]
        gender = row[5]
        if gender == "FALSE":
          break
        gender_map[uid] = gender
      rows_read += 1
  return gender_map

# generate train and test sets
def generate_two_split_dataset(input, labels):
  train = 0.7
  dev = 0.3
  x_train, x_test, y_train, y_test = train_test_split(input, labels, test_size=0.3, random_state=42)

  #save inputs and outputs
  np.save('train_input_counts.npy', x_train)

  np.save('train_labels.npy', y_train)

  np.save('test_input_counts.npy', x_test)

  np.save('test_labels.npy', y_test)
  np.savetxt('test_labels.txt', y_test)

  print("Completed split into train and test datasets!")

def process_command_line():
  """Sets command-line flags"""
  parser = argparse.ArgumentParser(description="Write and read formatted Json files")
  # optional arguments
  parser.add_argument('--bow', dest='is_bow', type=bool, default=False, help='Chooses to generate bag-of-words features')
  parser.add_argument('--grouped', dest='is_grouped', type=bool, default=False, help='Chooses to generate features grouped by (sender, recipient) pairs (as opposed to per-email features)')
  parser.add_argument('--thread', dest='is_thread', type=bool, default=False, help='Analyzes email text on the thread-level')
  parser.add_argument('--new_grouped_1', dest="is_grouped_1", type=bool, default=False, help='Alternating (sender, recipient) and (recipient, sender) groups')
  parser.add_argument('--new_grouped_2', dest="is_grouped_2", type=bool, default=False, help='Individualized (sender, recipient) and (recipient, sender) emails ')
  parser.add_argument('--new_grouped_3', dest="is_grouped_3", type=bool, default=False, help='Approach 1 all examples')
  parser.add_argument('--new_grouped_4', dest="is_grouped_4", type=bool, default=False, help='Approach 1 all examples individualized')
  parser.add_argument('--new_thread_1', dest='is_thread_1', type=bool, default=False, help='Analyzes email text on the thread-level - both (sender, recipient) and (recipient, sender) groups')
  parser.add_argument('--new_thread_2', dest='is_thread_2', type=bool, default=False, help='Analyzes email text on the thread-level - both (sender, recipient) and (recipient, sender) groups separated by individual email')
  parser.add_argument('--new_thread_3', dest='is_thread_3', type=bool, default=False, help='Analyzes email text on the thread-level - both (sender, recipient) and (recipient, sender) groups, handles no B->A email case')
  parser.add_argument('--new_thread_4', dest='is_thread_4', type=bool, default=False, help='Analyzes email text on the thread-level - both (sender, recipient) and (recipient, sender) groups separated by individual email, handles no B->A email case')
  args = parser.parse_args()
  return args

def main():
  args = process_command_line()

  populate_email_names_set()

  # Generates power labels and indices per thread
  if args.is_thread:
    with open('enron_database/emails_fixed.json') as file:
      d_emails = json.load(file)
    with open('enron_database/threads_fixed.json') as file:
      d_threads = json.load(file)
    thread_labels, thread_content = get_power_labels_and_indices_thread(d_threads, d_emails)
    print "Finished getting power labels and indices for thread."

    # Save thread contents
    thread_content = np.array(thread_content)
    np.save("thread_content.npy", thread_content)
    with open('thread_content.txt','wb') as f:
        np.savetxt(f, thread_content, delimiter='\n', fmt="%s")

    np.save("thread_labels.npy", thread_labels)
    np.savetxt("thread_labels.txt", thread_labels)

  if args.is_thread_1:
    with open('enron_database/emails_fixed.json') as file:
      d_emails = json.load(file)
    with open('enron_database/threads_fixed.json') as file:
      d_threads = json.load(file)
    thread_labels, thread_content_1, thread_content_2 = get_power_labels_and_indices_thread_1(d_threads, d_emails)
    print "Finished getting power labels and indices for thread."

    np.save("thread_labels_approach_1.npy", thread_labels)
    np.savetxt("thread_labels_approach_1.txt", thread_labels)

    # Save thread contents
    thread_content_1 = np.array(thread_content_1)
    np.save("thread_content_1.npy", thread_content_1)
    # with open('thread_content_1.txt','wb') as f:
    #     np.savetxt(f, thread_content_1, delimiter='\n', fmt="%s")

    thread_content_2 = np.array(thread_content_2)
    np.save("thread_content_2.npy", thread_content_2)
    with open('thread_content_2.txt','wb') as f:
        np.savetxt(f, thread_content_2, delimiter='\n', fmt="%s")

  if args.is_thread_2:
    with open('enron_database/emails_fixed.json') as file:
      d_emails = json.load(file)
    with open('enron_database/threads_fixed.json') as file:
      d_threads = json.load(file)
    thread_content_1, thread_content_2 = get_power_labels_and_indices_thread_2(d_threads, d_emails)
    print "Finished getting power labels and indices for thread."

    # Save thread contents
    thread_content_1 = np.array(thread_content_1)
    np.save("thread_content_1_individual.npy", thread_content_1)
    # with open('thread_content_1_individual.txt','wb') as f:
    #     np.savetxt(f, thread_content_1, delimiter='\n', fmt="%s")

    thread_content_2 = np.array(thread_content_2)
    np.save("thread_content_2_individual.npy", thread_content_2)
    with open('thread_content_2_individual.txt','wb') as f:
        np.savetxt(f, thread_content_2, delimiter='\n', fmt="%s")

  if args.is_thread_3:
    with open('enron_database/emails_fixed.json') as file:
      d_emails = json.load(file)
    with open('enron_database/threads_fixed.json') as file:
      d_threads = json.load(file)
    thread_labels, thread_content_1, thread_content_2 = get_power_labels_and_indices_thread_1(d_threads, d_emails, args.is_thread_3)
    print "Finished getting power labels and indices for thread."

    np.save("aug_data/approach1/thread_labels_approach_1_extended.npy", thread_labels)
    np.savetxt("thread_labels_approach_1_extended.txt", thread_labels)

    # Save thread contents
    thread_content_1 = np.array(thread_content_1)
    np.save("aug_data/approach1/thread_content_1_extended.npy", thread_content_1)
    # with open('thread_content_1.txt','wb') as f:
    #     np.savetxt(f, thread_content_1, delimiter='\n', fmt="%s")

    thread_content_2 = np.array(thread_content_2)
    np.save("aug_data/approach1/thread_content_2_extended.npy", thread_content_2)
    with open('thread_content_2_extended.txt','wb') as f:
        np.savetxt(f, thread_content_2, delimiter='\n', fmt="%s")

  if args.is_thread_4:
    with open('enron_database/emails_fixed.json') as file:
      d_emails = json.load(file)
    with open('enron_database/threads_fixed.json') as file:
      d_threads = json.load(file)
    labels, thread_content_1, thread_content_2 = get_power_labels_and_indices_thread_4(d_threads, d_emails)
    print "Finished getting power labels and indices for thread."

    np.save("thread_labels_extended.npy", labels)
    np.savetxt("thread_labels_extended.txt", labels)

    # Save thread contents
    thread_content_1 = np.array(thread_content_1)
    np.save("thread_content_1_individual_extended.npy", thread_content_1)
    # with open('thread_content_1_individual_extended.txt','wb') as f:
    #     np.savetxt(f, thread_content_1, delimiter='\n', fmt="%s")

    thread_content_2 = np.array(thread_content_2)
    np.save("thread_content_2_individual_extended.npy", thread_content_2)
    with open('thread_content_2_individual_extended.txt','wb') as f:
        np.savetxt(f, thread_content_2, delimiter='\n', fmt="%s")

   # Produces bag-of-words features
  if args.is_bow:
    with open('./enron_database/emails_fixed.json') as file:
      d_emails = json.load(file)

      if args.is_grouped:
        print "Generating *grouped* features for (sender, recipient) pairs..."
        # Generate features for (sender, recipient) pairs
        labels, email_contents, avgNumRecipients, avgNumTokensPerEmail = get_power_labels_and_indices_grouped(d_emails)
        print "Finished getting power labels and indices!"

        # save labels
        np.save('labels_grouped.npy', labels)
        np.savetxt('labels_grouped.txt', labels)

        # save email_contents
        email_contents = np.array(email_contents)
        np.save('email_contents_grouped.npy', email_contents)
        with open('email_contents_grouped.txt','wb') as f:
          np.savetxt(f, email_contents, delimiter='\n', fmt="%s")

        avgNumRecipients = np.array(avgNumRecipients)
        np.save('avg_num_recipients.npy', avgNumRecipients)
        np.savetxt('avg_num_recipients.txt', avgNumRecipients)

        avgNumTokensPerEmail = np.array(avgNumTokensPerEmail)
        np.save('avg_num_tokens_per_email.npy', avgNumTokensPerEmail)
        np.savetxt('avg_num_tokens_per_email.txt', avgNumTokensPerEmail)

        print "Finished saving email_contents and labels to files!"

      elif args.is_grouped_1:
        print "Generating *grouped* features for (sender, recipient) and (recipient, sender) pairs..."
        # Generate features for (sender, recipient) pairs
        labels, email_contents_1, email_contents_2, avgNumRecipients_1, avgNumRecipients_2, avgNumTokensPerEmail_1, avgNumTokensPerEmail_2, avgNumEmailsPerGroup_1, avgNumEmailsPerGroup_2 = get_power_labels_and_indices_grouped_1(d_emails, args.is_grouped_3)
        print "Finished getting power labels and indices!"

        # save labels
        np.save('labels_grouped_approach_1.npy', labels)
        np.savetxt('labels_grouped_approach_1.txt', labels)
        exit(0)

        avgNumRecipients_1 = np.array(avgNumRecipients_1)
        np.save('avg_num_recipients_1.npy', avgNumRecipients_1)
        np.savetxt('avg_num_recipients_1.txt', avgNumRecipients_1)

        avgNumRecipients_2 = np.array(avgNumRecipients_2)
        np.save('avg_num_recipients_2.npy', avgNumRecipients_2)
        np.savetxt('avg_num_recipients_2.txt', avgNumRecipients_2)

        avgNumTokensPerEmail_1 = np.array(avgNumTokensPerEmail_1)
        np.save('avg_num_tokens_per_email_1.npy', avgNumTokensPerEmail_1)
        np.savetxt('avg_num_tokens_per_email_1.txt', avgNumTokensPerEmail_1)

        avgNumTokensPerEmail_2 = np.array(avgNumTokensPerEmail_2)
        np.save('avg_num_tokens_per_email_2.npy', avgNumTokensPerEmail_2)
        np.savetxt('avg_num_tokens_per_email_2.txt', avgNumTokensPerEmail_2)

        # save email_contents
        email_contents_1 = np.array(email_contents_1)
        np.save('email_contents_grouped_1.npy', email_contents_1)
        # with open('email_contents_grouped_1.txt','wb') as f:
        #   np.savetxt(f, email_contents_1, delimiter='\n', fmt="%s")

        email_contents_1 = np.array(email_contents_2)
        np.save('email_contents_grouped_2.npy', email_contents_2)
        with open('email_contents_grouped_2.txt','wb') as f:
          np.savetxt(f, email_contents_2, delimiter='\n', fmt="%s")
        print "Finished saving email_contents and labels to files!"

      elif args.is_grouped_2:
        print "Generating *grouped and individualized* features for (sender, recipient) and (recipient, sender) pairs..."
        # Generate features for (sender, recipient) pairs
        email_contents_1, email_contents_2 = get_power_labels_and_indices_grouped_2(d_emails)
        print "Finished getting power labels and indices!"

        # save email_contents
        email_contents = np.array(email_contents_1)
        # np.save('email_contents_grouped_1_individual.npy', email_contents_1)
        # with open('email_contents_grouped_1_individual.txt','wb') as f:
        #   np.savetxt(f, email_contents_1, delimiter='\n', fmt="%s")

        email_contents = np.array(email_contents_2)
        np.save('email_contents_grouped_2_individual.npy', email_contents_2)
        with open('email_contents_grouped_2_individual.txt','wb') as f:
          np.savetxt(f, email_contents_2, delimiter='\n', fmt="%s")
        print "Finished saving email_contents and labels to files!"

      elif args.is_grouped_3:
        print "Generating *grouped and individualized* features for (sender, recipient) and (recipient, sender) pairs..."
        # Generate features for (sender, recipient) pairs
        labels, email_contents_1, email_contents_2, avgNumRecipients_1, avgNumRecipients_2, avgNumTokensPerEmail_1, avgNumTokensPerEmail_2, avgNumEmailsPerGroup_1, avgNumEmailsPerGroup_2 = get_power_labels_and_indices_grouped_1(d_emails, args.is_grouped_3)
        print "Finished getting power labels and indices!"

        # save labels
        np.save('labels_grouped_approach_1_extended.npy', labels)
        np.savetxt('labels_grouped_approach_1_extended.txt', labels)

        avgNumRecipients_1 = np.array(avgNumRecipients_1)
        np.save('avg_num_recipients_extended_1.npy', avgNumRecipients_1)
        np.savetxt('avg_num_recipients_1_extended.txt', avgNumRecipients_1)

        avgNumRecipients_2 = np.array(avgNumRecipients_2)
        np.save('avg_num_recipients_extended_2.npy', avgNumRecipients_2)
        np.savetxt('avg_num_recipients_2_extended.txt', avgNumRecipients_2)

        avgNumTokensPerEmail_1 = np.array(avgNumTokensPerEmail_1)
        np.save('avg_num_tokens_per_email_extended_1.npy', avgNumTokensPerEmail_1)
        np.savetxt('avg_num_tokens_per_email_1_extended.txt', avgNumTokensPerEmail_1)

        avgNumTokensPerEmail_2 = np.array(avgNumTokensPerEmail_2)
        np.save('avg_num_tokens_per_email_extended_2.npy', avgNumTokensPerEmail_2)
        np.savetxt('avg_num_tokens_per_email_2_extended.txt', avgNumTokensPerEmail_2)

        # save email_contents
        email_contents_1 = np.array(email_contents_1)
        np.save('email_contents_grouped_1_extended.npy', email_contents_1)
        # with open('email_contents_grouped_1_extended.txt','wb') as f:
        #   np.savetxt(f, email_contents_1, delimiter='\n', fmt="%s")

        email_contents_2 = np.array(email_contents_2)
        np.save('email_contents_grouped_2_extended.npy', email_contents_2)
        with open('email_contents_grouped_2_extended.txt','wb') as f:
          np.savetxt(f, email_contents_2, delimiter='\n', fmt="%s")
        print "Finished saving email_contents and labels to files!"
      elif args.is_grouped_4:
        print "Generating *grouped and individualized* features for (sender, recipient) and (recipient, sender) pairs..."
        # Generate features for (sender, recipient) pairs
        email_contents_1, email_contents_2 = get_power_labels_and_indices_grouped_2(d_emails, args.is_grouped_4)
        print "Finished getting power labels and indices!"

        # save email_contents
        email_contents_1 = np.array(email_contents_1)
        np.save('aug_data/approach2/email_contents_grouped_1_individual_extended.npy', email_contents_1)
        # with open('email_contents_grouped_1_individual_extended.txt','wb') as f:
        #   np.savetxt(f, email_contents_1, delimiter='\n', fmt="%s")

        email_contents_2 = np.array(email_contents_2)
        np.save('aug_data/approach2/email_contents_grouped_2_individual_extended.npy', email_contents_2)
        with open('email_contents_grouped_2_individual_extended.txt','wb') as f:
          np.savetxt(f, email_contents_2, delimiter='\n', fmt="%s")
        print "Finished saving email_contents and labels to files!"
      else:
        labels, email_contents, num_recipients_vector, gender_feature_vector = get_power_labels_and_indices(d_emails)
        print "Finished getting power labels and indices!"

        #save extra features to files
        num_recipients_features = np.array(num_recipients_vector)
        np.save('num_recipients_features_nodup.npy', num_recipients_features)
        np.savetxt('num_recipients_features_nodup.txt', num_recipients_features)

        gender_features = np.array(gender_feature_vector)
        np.save('gender_features_nodup.npy', gender_features)

        # save email_contents and labels to files
        email_contents = np.array(email_contents)
        np.save('email_contents_nodup.npy', email_contents)
        with open('email_contents_nodup.txt','wb') as f:
          np.savetxt(f, email_contents, delimiter='\n', fmt="%s")

        np.save('labels_nodup.npy', labels)
        np.savetxt('labels_nodup.txt', labels)

        print "Finished saving email_contents and labels to files!"
        exit(0)

        # transform email_contents to sparse vectors of word counts
        count_vect = CountVectorizer()
        all_input_counts = count_vect.fit_transform(email_contents)
        print "all input vectors shape:", all_input_counts.shape
        print "all_input_counts counts:", all_input_counts[:5]
        np.save('all_input_counts.npy', all_input_counts)

        generate_two_split_dataset(all_input_counts, labels)

    print "Completed labels and feature vectors!"

if __name__ == "__main__":
    main()