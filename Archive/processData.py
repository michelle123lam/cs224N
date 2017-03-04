import bson
import json
import argparse

# Loads and processes the Enron Organizational Hierarchy json data
# CS 224N Winter 2017

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


def main():
  args = process_command_line()

  if args.write_json:
    writeJson('enron_database/emails.json', 'enron_database/emails_fixed.json')
    writeJson('enron_database/entities.json', 'enron_database/entities_fixed.json')
    writeJson('enron_database/threads.json', 'enron_database/threads_fixed.json')
  if args.read_json:
    readJson('enron_database/emails_fixed.json')
    readJson('enron_database/entities_fixed.json')
    readJson('enron_database/threads_fixed.json')


if __name__ == "__main__":
    main()