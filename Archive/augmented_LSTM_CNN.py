"""
augmented_LSTM.py
-------
Approach 1:
Separate A's emails and B's emails
LSTM on each party's emails
Non-lex features combined with LSTM and merged to get final classification

Commands:
Generate Approach 1 data: 
	python ./augmented_LSTM_CNN.py --prepareData=True --approach=1

Run Approach 1 LSTM:
	python ./augmented_LSTM_CNN.py --approach=1 --model=LSTM
"""
import argparse
import numpy as np
import pickle
# from keras.models import Sequential
# from keras.layers import Merge, Activation, Dense
# from keras.layers.recurrent import LSTM

"""
def getPreds(merged_a, merged_b, data, batch_size, num_epochs):
	# TODO
	# merge a and b models
	merged_model = Sequential()
	merged_model.add(Merge([merged_a, merged_b], mode='concat'))

	# softmax classification
	merged_model.add(Dense(1, activation='softmax'))
	# merged_model.add(Activation('softmax'))
	merged_model.compile(optimizer='adam', loss='binary_crossentropy')
	merged_model.fit(data['train']['x'], data['train']['y'],
		batch_size=batch_size,
		epochs=num_epochs,
		validation_data=(data['dev']['x'], data['dev']['y']))
	return merged_model

def getNonLexicalFeats(feats_raw):
	# TODO

def getLSTM(emails, max_email_len, dropout=0.8):
	# TODO
	# Take in email text
	n_emails = len(emails)
	email_len = max_email_len
	output_dim = 20

	model = Sequential()
	model.add(LSTM(output_dim, input_shape=(n_emails, email_len), dropout=droput))
	return model

	# EXAMPLE:
	# first_model = Sequential()
	# first_model.add(LSTM(output_dim, input_shape=(m, input_dim)))

	# second_model = Sequential()
	# second_model.add(LSTM(output_dim, input_shape=(n-m, input_dim)))

	# model = Sequential()
	# model.add(Merge([first_model, second_model], mode='concat'))
	# model.add(Dense(1))
	# model.add(Activation('sigmoid'))
	# model.compile(optimizer='RMSprop', loss='binary_crossentropy')
	# model.fit([X[:,:m,:], X[:,m:,:]], y)


def AugLSTM1_decomposed(a_data, b_data):
	# TODO: get a_emails, b_emails (pad sequences to max_email_len); get a_feats, b_feats
	max_email_len = # TODO
	LSTM_a = getLSTM(a_emails, max_email_len)
	LSTM_b = getLSTM(b_emails, max_email_len)

	nonLex_a = getNonLexicalFeats(a_feats)
	nonLex_b = getNonLexicalFeats(b_feats)

	merged_a = # concatenate LSTM_a and nonLex_a
	merged_b = # concatenate LSTM_b and nonLex_b

	data = # train/dev/test data; format = data['train']['x'] etc.
	batch_size = # TODO
	num_epochs = # TODO

	final_model = getPreds(merged_a, merged_b, data, batch_size, num_epochs)

	score, acc = final_model.evaluate(data['dev']['x'], data['dev']['x'], batch_size=batch_size)
"""

# AugLSTM: Approach 1
def AugLSTM1_full(data, output_dim=20, dropout=0.8, batch_size=30, num_epochs=10):
	# Create LSTMs
	LSTM_a = Sequential()
	LSTM_a.add(LSTM(output_dim, input_shape=(metadata['n_emails_a'], metadata['max_email_len']), dropout=dropout))
	LSTM_b = Sequential()
	LSTM_b.add(LSTM(output_dim, input_shape=(metadata['n_emails_b'], metadata['max_email_len']), dropout=dropout))

	# Merge non-lexical features
	# TODO: get non-lexical features

	# Merge LSTMs
	merged_model = Sequential()
	merged_model.add(Merge([LSTM_a, LSTM_b], mode='concat'))

	# Softmax classification
	merged_model.add(Dense(1, activation='softmax'))
		# merged_model.add(Activation('softmax'))
	merged_model.compile(optimizer='adam', loss='binary_crossentropy')
	print "Compiled merged_model!"

	# Fit model
	# TODO: how to feed the data into the correct sides of LSTM??
	# model.fit([X[:,:m,:], X[:,m:,:]], y)
	merged_model.fit([data['train']['x'][0], data['train']['x'][1]], data['train']['y'],
		batch_size=batch_size,
		epochs=num_epochs,
		validation_data=(data['dev']['x'], data['dev']['y']))
	print "Fitted merged_model!"

	# Evaluate model
	score, acc = final_model.evaluate(data['dev']['x'], data['dev']['x'], batch_size=batch_size)
	print "score:", score, " acc:", acc


# AugCNN: Approach 1
def AugCNN1_full(data, output_dim=20, dropout=0.8, batch_size=30, num_epochs=10):
	# TODO
	return


"""
Process data for Approach 1
- Reads in raw email text and labels
- Shuffles data in random order
- Splits data into train/dev/test
- Saves data to pkl

split_data dict --> 
	keys: ['train' | 'dev' | 'test'] --> 
		keys: ['x' | 'y'] --> 
			arr of [a_value, b_value]
"""
def processData1(raw_x_file, raw_y_file, pkl_file):
	data = {}
	data['x'] = []
	data['y'] = []

	# Load txt file
	with open(raw_x_file, 'r') as f:
		raw_x = f.readlines()
	with open(raw_y_file, 'r') as f:
		raw_y = f.readlines()

	# Separate A's and B's emails for each pairing
	n_pairs = len(raw_x) / 2

	for pair_i in range(n_pairs):
		a_ind = pair_i * 2
		b_ind = a_ind + 1
		data['x'].append([raw_x[a_ind], raw_x[b_ind]])
		data['y'].append([raw_y[a_ind], raw_y[b_ind]])
	print "Read in data!"

	# Shuffle data
	np.random.seed(10)
	shuffle_indices = np.random.permutation(np.arange(n_pairs))
	data['x'] = [data['x'][i] for i in shuffle_indices]
	data['y'] = [data['y'][i] for i in shuffle_indices]
	print "Shuffled data!"

	# Split into train/dev/test
	train = 0.6
	dev = 0.2
	test = 0.2
	train_cutoff = int(0.6 * n_pairs)
	dev_cutoff = int(0.8 * n_pairs)
	test_cutoff = n_pairs

	split_data = {
		'train': {},
		'dev': {},
		'test': {},
	}
	split_data['train']['x'] = [data['x'][i] for i in range(train_cutoff)]
	split_data['dev']['x'] = [data['x'][i] for i in range(train_cutoff, dev_cutoff)]
	split_data['test']['x'] = [data['x'][i] for i in range(dev_cutoff, test_cutoff)]
	split_data['train']['y'] = [data['y'][i] for i in range(train_cutoff)]
	split_data['dev']['y'] = [data['y'][i] for i in range(train_cutoff, dev_cutoff)]
	split_data['test']['y'] = [data['y'][i] for i in range(dev_cutoff, test_cutoff)]
	print "Split data!"

	# Save shuffled and split data to pickle
	with open(pkl_file, 'wb') as f:
		pickle.dump(split_data, f)
	print "Saved data!"

	return split_data

def main(args):
	raw_x_file = 'aug_data/grouped_test_x.txt'
	raw_y_file = 'aug_data/grouped_test_y.txt'
	pkl_file = 'aug_data/grouped_test.pkl'

	if args.prepareData:
		# Prepare train/dev/test data

		# Approach 1 data
		if args.approach == 1:
			print "Preparing Approach 1 data!"
			data = processData1(raw_x_file, raw_y_file, pkl_file)

	else:
		# Run model
		with open(pkl_file, 'rb') as f:
		    data = pickle.load(f)

	    # Approach 1
		if args.approach == 1:
			if args.model == 'LSTM':
				print "data:", data
				# print "Running Approach 1 LSTM!"
				# AugLSTM1_full(data,
				# 	batch_size=5,
				# 	num_epochs=3)

			elif args.model == 'CNN':
				print "Running Approach 1 CNN!"
				AugCNN1_full(data,
					batch_size=5,
					num_epochs=3)

		# Approach 2
		elif args.approach == 2:
			# TODO
			return


if __name__ == "__main__":
	# Prepare command line flags
	parser = argparse.ArgumentParser(description='Run augmented LSTM or CNN models')
	parser.add_argument('--prepareData', type=bool, default=False, help='prepare data for the model (default=False)')
	parser.add_argument('--approach', type=int, default=1, help="which number approach to use (default=1)")
	parser.add_argument('--model', type=str, default='LSTM', help="which model to use (default=LSTM)")

	args = parser.parse_args()
	main(args)
