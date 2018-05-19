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

from keras.models import Sequential, Model
from keras.layers import Embedding, Input, Conv1D, Conv2D, MaxPool2D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Flatten, Dropout
from keras.layers.core import Dense
from keras.layers.merge import concatenate
from keras.layers.recurrent import LSTM

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from nltk import tokenize

# Metrics wrapper
class Metrics(Callback):
	def on_train_begin(self, logs={}):
		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []

	def on_epoch_end(self, epoch, logs={}):
		val_predict = (np.asarray(self.model.predict([self.validation_data[0], self.validation_data[1]]))).round()
		val_targ = self.validation_data[2]
		_val_f1 = f1_score(val_targ, val_predict)
		_val_recall = recall_score(val_targ, val_predict)
		_val_precision = precision_score(val_targ, val_predict)
		self.val_f1s.append(_val_f1)
		self.val_recalls.append(_val_recall)
		self.val_precisions.append(_val_precision)
		print " - val_f1: %f - val_precision: %f - val_recall %f" %(_val_f1, _val_precision, _val_recall)
		return

metrics = Metrics()

# AugLSTM: Approach 1
def AugLSTM1_full(data, output_dim=100, dropout=0.2, batch_size=30, num_epochs=10, max_email_words=50, word_vec_dim=100):
	input_shape=(max_email_words, word_vec_dim)

	# Create LSTMs
	input_a = Input(shape=input_shape, dtype='float32')
	LSTM_a = LSTM(output_dim, input_shape=input_shape, dropout=dropout)(input_a)

	input_b = Input(shape=input_shape, dtype='float32')
	LSTM_b = LSTM(output_dim, input_shape=input_shape, dropout=dropout)(input_b)

	# Merge non-lexical features
	# TODO: get non-lexical features

	# Merge LSTMs
	merged = concatenate([LSTM_a, LSTM_b])
	
	# Softmax classification
	dense_out = Dense(1, activation='sigmoid')(merged)
	merged_model = Model(inputs=[input_a, input_b], outputs=[dense_out])
	merged_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	print "Compiled merged_model!"

	# Fit model
	merged_model.fit(
		[np.array(data['train']['x'])[:,0,:,:], 
		np.array(data['train']['x'])[:,1,:,:]], 
		np.array(data['train']['y']),
		batch_size=batch_size,
		epochs=num_epochs,
		validation_data=(
			[np.array(data['dev']['x'])[:,0,:,:], 
			np.array(data['dev']['x'])[:,1,:,:]], 
			np.array(data['dev']['y'])),
		callbacks=[metrics])
	print "Fitted merged_model!"

	# Evaluate model
	val_predict = np.asarray(merged_model.predict(
		[np.array(data['test']['x'])[:,0,:,:],
		np.array(data['test']['x'])[:,1,:,:]])).round()
	val_target = np.array(data['test']['y'])
	val_f1 = f1_score(val_target, val_predict)
	val_recall = recall_score(val_target, val_predict)
	val_precision = precision_score(val_target, val_predict)
	val_acc = float(np.mean(val_predict == val_target))
	print " - val_acc: %f - val_f1: %f - val_precision: %f - val_recall %f" %(val_acc, val_f1, val_precision, val_recall)

def get_CNN(num_filters, strides, activation, max_email_words, word_vec_dim):
	input_shape=(max_email_words, word_vec_dim, 1)
	# input_shape=(max_email_words,)
	cur_input = Input(shape=input_shape, dtype='float32')
	
	# 1) Simple Conv1D version
	# conv = Conv1D(num_filters, kernel_size=5, strides=1, activation=activation)(cur_input)
	# l_cov1= Conv1D(128, 2, activation='relu')(cur_input)
	# l_pool1 = MaxPooling1D(4)(l_cov1)
	# l_cov2 = Conv1D(128, 2, activation='relu')(l_pool1)
	# l_pool2 = MaxPooling1D(2)(l_cov2)
	# l_cov3 = Conv1D(128, 2, activation='relu')(l_pool2)
	# l_pool3 = MaxPooling1D(2)(l_cov3)  # global max pooling
	# flatten = Flatten()(l_pool3)

	# 2) Simple Conv2D version

	# 3) Multiple filters version
	filter_sizes = [3,4,5]
	conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], word_vec_dim), strides=strides, activation=activation, padding='valid')(cur_input)
	conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], word_vec_dim), strides=strides, activation=activation, padding='valid')(cur_input)
	conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], word_vec_dim), strides=strides, activation=activation, padding='valid')(cur_input)
	maxpool_0 = MaxPool2D(pool_size=(max_email_words - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
	maxpool_1 = MaxPool2D(pool_size=(max_email_words - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
	maxpool_2 = MaxPool2D(pool_size=(max_email_words - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)
	concatenated_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
	flatten = Flatten()(concatenated_tensor)

	# CNN = Sequential()
	# CNN.add(Embedding(input_dim=word_vec_dim, output_dim=output_dim, input_length=max_email_words))
	# CNN.add(Conv2D(output_dim, kernel_size=kernel_size, strides=strides, input_shape=input_shape, activation=activation))
	# CNN.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	# CNN.add(Conv2D(64, (5, 5), activation='relu'))
	# CNN.add(MaxPooling2D(pool_size=(2, 2)))
	# CNN.add(Flatten())
	# CNN.add(Dense(1000, activation='relu'))
	# CNN.add(Dense(num_classes, activation='softmax'))
	return cur_input, flatten

# AugCNN: Approach 1
def AugCNN1_full(data, num_filters=32, batch_size=30, num_epochs=10, strides=(1, 1), activation='relu', max_email_words=50, word_vec_dim=100, dropout=0.2):
	
	# Create CNNs
	input_a, CNN_a = get_CNN(num_filters, strides, activation, max_email_words=max_email_words, word_vec_dim=word_vec_dim)
	input_b, CNN_b = get_CNN(num_filters, strides, activation, max_email_words=max_email_words, word_vec_dim=word_vec_dim)

	# Merge non-lexical features
	# TODO: get non-lexical features

	# Merge CNNs
	merged = concatenate([CNN_a, CNN_b])
	dropout = Dropout(dropout)(merged) # dropout = fraction of input units to drop
	
	# Softmax classification
	dense_out = Dense(1, activation='sigmoid')(dropout)
	merged_model = Model(inputs=[input_a, input_b], outputs=[dense_out])
	merged_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	print "Compiled merged_model!"

	# Fit model
	merged_model.fit(
		[np.array(data['train']['x'])[:,0,:,:,np.newaxis], 
		np.array(data['train']['x'])[:,1,:,:,np.newaxis]], 
		np.array(data['train']['y']),
		batch_size=batch_size,
		epochs=num_epochs,
		validation_data=(
			[np.array(data['dev']['x'])[:,0,:,:,np.newaxis], 
			np.array(data['dev']['x'])[:,1,:,:,np.newaxis]], 
			np.array(data['dev']['y'])),
			callbacks=[metrics])
	print "Fitted merged_model!"

	# Evaluate model
	val_predict = np.asarray(merged_model.predict(
		[np.array(data['test']['x'])[:,0,:,:,np.newaxis],
		np.array(data['test']['x'])[:,1,:,:,np.newaxis]])).round()
	val_target = np.array(data['test']['y'])
	val_f1 = f1_score(val_target, val_predict)
	val_recall = recall_score(val_target, val_predict)
	val_precision = precision_score(val_target, val_predict)
	val_acc = float(np.mean(val_predict == val_target))
	print " - val_acc: %f - val_f1: %f - val_precision: %f - val_recall %f" %(val_acc, val_f1, val_precision, val_recall)

"""
Vectorizes email text data
- TODO: decide best way to capture email in terms of word vecs:
	- [currently] Sequence of word vectors (capped at max_email_words)
	OR
	- Sum of sentence vectors (each sentence capped at max_sentence_words) (what we did before)
	- Paragraph vectors
"""
def get_vectorized_email(email, wordVectors, max_email_words=50, word_vec_dim=100):
	result = np.zeros((max_email_words, word_vec_dim))
	words = tokenize.word_tokenize(email.decode('utf-8'))
	# print "words:", words
	words = words[:max_email_words]
	wv = np.array([wordVectors[word] for word in words if word in wordVectors])
	if (wv.shape[0] > 0):
		result[:wv.shape[0], :wv.shape[1]] = wv
	return result


"""
Process data for Approach 1
- Reads in raw email text and labels
- Shuffles data in random order
- Splits data into train/dev/test
- Saves data to pkl

split_data dict --> 
	keys: ['train' | 'dev' | 'test'] --> 
		keys: ['x' | 'y'] --> 
			for 'x': [a_value, b_value]
			for 'y': label (relation from a-to-b)

x_values are a list of up to [max_email_words] word vectors (each 100 dimensions)
"""
def processData1(raw_xa_file, raw_xb_file, raw_y_file, pkl_file):
	data = {}
	data['x'] = []
	data['y'] = []

	# Load word vectors and tokens
	with open('aug_data/glove.6B.100d.txt', 'r') as f:
		wordVectors = {}
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			wordVectors[word] = coefs
	print "Loaded GloVe!"

	# Load npy file
	raw_xa = np.load(raw_xa_file)
	raw_xb = np.load(raw_xb_file)
	raw_y = np.load(raw_y_file)

	"""
	# Load txt file
	with open(raw_x_file, 'r') as f:
		raw_x = f.readlines()
	with open(raw_y_file, 'r') as f:
		raw_y = f.readlines()
	"""

	# Separate A's and B's emails for each pairing
	n_pairs = len(raw_xa)

	for pair_i in range(n_pairs):
		a_email = get_vectorized_email(raw_xa[pair_i], wordVectors)
		b_email = get_vectorized_email(raw_xb[pair_i], wordVectors)
		data['x'].append([a_email, b_email])
		data['y'].append([raw_y[pair_i]])
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

	# Save shuffled, split data to pickle
	with open(pkl_file, 'wb') as f:
		print "split_data:", split_data
		pickle.dump(split_data, f)
	print "Saved data!"

	return split_data

def main(args):
	# TODO: update to true data file
	raw_xa_file = 'aug_data/approach1/email_contents_grouped_1.npy'
	raw_xb_file = 'aug_data/approach1/email_contents_grouped_2.npy'
	raw_y_file = 'aug_data/approach1/labels_grouped.npy'
	pkl_file = 'aug_data/approach1/grouped.pkl'

	# pkl_file = 'aug_data/approach1_toy/grouped_test.pkl'

	# Prepare train/dev/test data
	if args.prepareData:
		# Approach 1 data
		if args.approach == 1:
			print "Preparing Approach 1 data!"
			data = processData1(raw_xa_file, raw_xb_file, raw_y_file, pkl_file)

		# Approach 2 data
		elif args.approach == 2:
			print "Preparing Approach 2 data!"
			return

	# Run model
	else:
		"""
		data -->
			keys: ['train'|'dev'|'test'] -->
				keys: ['x', 'y'] -->
					for 'x': [a_value, b_value,
						where a_value has shape (max_email_words=50, word_vec_dim=100)
					for 'y': label (relation from a-to-b)
		"""
		with open(pkl_file, 'rb') as f:
		    data = pickle.load(f)

	    # Approach 1 model
		if args.approach == 1:
			if args.model == 'LSTM':
				print "Running Approach 1 LSTM!"
				print "trainX shape:", np.shape(data['train']['x'])
				print "trainY shape:", np.shape(data['train']['y'])
				print "devX shape:", np.shape(data['dev']['x'])
				print "sliced trainX shape:", np.shape(np.array(data['train']['x'])[:,0,:,:])
				AugLSTM1_full(data,
					batch_size=30,
					num_epochs=10)

			elif args.model == 'CNN':
				print "Running Approach 1 CNN!"
				AugCNN1_full(data,
					batch_size=2,
					num_epochs=10)

		# Approach 2 model
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
