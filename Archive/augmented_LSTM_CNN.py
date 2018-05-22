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
import tensorflow as tf
import random as rn
import pickle

from keras.models import Sequential, Model
from keras.layers import Embedding, Input, Conv1D, Conv2D, MaxPool2D, MaxPooling1D, GlobalMaxPooling1D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import Flatten, Dropout
from keras.layers.core import Dense
from keras.layers.merge import concatenate
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import Callback
from keras import backend as K
from keras import regularizers
import os
os.environ['PYTHONHASHSEED'] = '0'

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# from hyperas import optim
# from hyperas.distributions import choice, uniform, conditional
# from hyperopt import Trials, STATUS_OK, rand, tpe

from nltk import tokenize

# Set Theano backend
# def set_keras_backend(backend):

#     if K.backend() != backend:
#         os.environ['KERAS_BACKEND'] = backend
#         reload(K)
#         assert K.backend() == backend

# set_keras_backend("theano")

# Random seed
np.random.seed(1)
tf.set_random_seed(2)
rn.seed(3)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

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


# AugLSTM: Approach 1 ---------------------------------
def AugLSTM1_full_data():
	pkl_file = 'aug_data/approach1/grouped_100.pkl'
	with open(pkl_file, 'rb') as f:
		data = pickle.load(f)
	output_dim = 100
	dropout = 0.3 # Tuned!
	batch_size=30
	num_epochs=50
	max_email_words=100
	word_vec_dim=100
	use_non_lex=True
	return data, output_dim, dropout, batch_size, num_epochs, max_email_words, word_vec_dim, use_non_lex

"""
Previous hyperparam tunes:
# epochs={{choice([10, 20, 50])}},
"""
def AugLSTM1_full(data, output_dim=100, dropout=0.2, batch_size=30, num_epochs=10, max_email_words=50, word_vec_dim=100, use_non_lex=True):
	input_shape=(max_email_words, word_vec_dim)

	# Create LSTMs
	input_a = Input(shape=input_shape, dtype='float32')
	LSTM_a = LSTM(output_dim, input_shape=input_shape, dropout=dropout)(input_a)

	input_b = Input(shape=input_shape, dtype='float32')
	LSTM_b = LSTM(output_dim, input_shape=input_shape, dropout=dropout)(input_b)

	if use_non_lex:
		# Merge non-lexical features with each LSTM
		# Prepare non-lexical data
		data_nonlex, input_nonlex_a, input_nonlex_b, non_lex_features_a, non_lex_features_b = get_nonlex(data)

		# Merge LSTMs
		merged = concatenate([LSTM_a, non_lex_features_a, LSTM_b, non_lex_features_b])
		merged_dense = Dense(32, activation='relu')(merged)
	
		# Softmax classification
		dense_out = Dense(1, activation='sigmoid')(merged_dense)
		merged_model = Model(inputs=[input_a, input_nonlex_a, input_b, input_nonlex_b], outputs=[dense_out])

		# Prepare input data
		print "nl:", np.shape(np.array(data_nonlex['dev'])[0])
		print "x:", np.shape(np.array(data['dev']['x'])[:,0,:,:])
		print "y:", np.shape(np.array(data['dev']['y']))
		x_train = [np.array(data['train']['x'])[:,0,:,:],
			data_nonlex['train'][0], 
			np.array(data['train']['x'])[:,1,:,:], 
			data_nonlex['train'][1]]
		x_dev = [np.array(data['dev']['x'])[:,0,:,:], 
			data_nonlex['dev'][0],
			np.array(data['dev']['x'])[:,1,:,:], 
			data_nonlex['dev'][1]]
		x_test = [np.array(data['test']['x'])[:,0,:,:],
			data_nonlex['test'][0],
			np.array(data['test']['x'])[:,1,:,:],
			data_nonlex['test'][1]]

	else:
		# Just use lexical features
		# Merge LSTMs
		merged = concatenate([LSTM_a, LSTM_b])
		merged_dense = Dense(32, activation='relu')(merged)
	
		# Softmax classification
		dense_out = Dense(1, activation='sigmoid')(merged_dense)
		merged_model = Model(inputs=[input_a, input_b], outputs=[dense_out])

		# Prepare input data
		x_train = [np.array(data['train']['x'])[:,0,:,:],
			np.array(data['train']['x'])[:,1,:,:]]
		x_dev = [np.array(data['dev']['x'])[:,0,:,:], 
			np.array(data['dev']['x'])[:,1,:,:]]
		x_test = [np.array(data['test']['x'])[:,0,:,:],
			np.array(data['test']['x'])[:,1,:,:]]

	merged_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	print("Compiled merged_model!")


	# Fit model
	merged_model.fit(x_train, np.array(data['train']['y']),
		batch_size=batch_size,
		# epochs=num_epochs,
		epochs={{choice([10, 20, 50])}},
		validation_data=(x_dev, np.array(data['dev']['y'])),
		# callbacks=[metrics]
		)
	print "Fitted merged_model!"

	# Evaluate model
	test_predict = np.asarray(merged_model.predict(x_test)).round()
	test_target = np.array(data['test']['y'])
	test_f1 = f1_score(test_target, test_predict)
	test_recall = recall_score(test_target, test_predict)
	test_precision = precision_score(test_target, test_predict)
	test_acc = float(np.mean(test_predict == test_target))
	print " - test_acc: %f - test_f1: %f - test_precision: %f - test_recall %f" %(test_acc, test_f1, test_precision, test_recall)

	return {'loss': -test_f1, 'status': STATUS_OK, 'model': merged_model}

# AugLSTM: Approach 2 ---------------------------------
def AugLSTM2_full_data():
	pkl_file = 'aug_data/approach2/grouped_100.pkl'
	with open(pkl_file, 'rb') as f:
		data = pickle.load(f)
	output_dim = 100
	dropout = 0.3
	batch_size=30 # -
	num_epochs=50 # -
	max_email_words=100
	word_vec_dim=100
	use_non_lex=True
	max_emails=5
	return data, output_dim, dropout, batch_size, num_epochs, max_email_words, word_vec_dim, use_non_lex, max_emails

def get_approach2_lex_input(data, split, max_emails, isCNN=False):
	lex_input = []
	for person_i in range(2):
		for email_i in range(max_emails):
			if isCNN:
				cur_email = np.array(data[split]['x'])[:,person_i,email_i,:,:,np.newaxis]
			else:
				cur_email = np.array(data[split]['x'])[:,person_i,email_i,:,:]
			
			lex_input.append(cur_email)
	return lex_input

def get_LSTM(input_shape, output_dim, dropout):
	cur_input = Input(shape=input_shape, dtype='float32')
	cur_LSTM = LSTM(output_dim, input_shape=input_shape, dropout=dropout)(cur_input)
	return cur_input, cur_LSTM

def AugLSTM2_full(data, output_dim=100, dropout=0.2, batch_size=30, num_epochs=10, max_email_words=50, word_vec_dim=100, use_non_lex=True, max_emails=5):
	input_shape=(max_email_words, word_vec_dim)

	# Create LSTMs
	inputs_a = []
	inputs_b = []
	LSTMs_a = []
	LSTMs_b = []
	for i in range(max_emails):
		input_a, LSTM_a = get_LSTM(input_shape, output_dim, dropout) # Tuning
		input_b, LSTM_b = get_LSTM(input_shape, output_dim, dropout) # Tuning
		inputs_a.append(input_a)
		inputs_b.append(input_b)
		LSTMs_a.append(LSTM_a)
		LSTMs_b.append(LSTM_b)

	if use_non_lex:
		# Merge non-lexical features with each LSTM
		# Prepare non-lexical data
		data_nonlex, input_nonlex_a, input_nonlex_b, non_lex_features_a, non_lex_features_b = get_nonlex(data)

		# Merge LSTMs
		LSTMs_and_nonlex_ab = LSTMs_a + LSTMs_b
		LSTMs_and_nonlex_ab.append(non_lex_features_a)
		LSTMs_and_nonlex_ab.append(non_lex_features_b)
		inputs_and_nonlex_ab = inputs_a + inputs_b
		inputs_and_nonlex_ab.append(input_nonlex_a)
		inputs_and_nonlex_ab.append(input_nonlex_b)
		merged = concatenate(LSTMs_and_nonlex_ab)
		merged_dense = Dense(32, activation='relu')(merged)
	
		# Softmax classification
		dense_out = Dense(1, activation='sigmoid')(merged_dense)
		merged_model = Model(inputs=inputs_and_nonlex_ab, outputs=[dense_out])

		# Prepare input data
		# print "nl:", np.shape(np.array(data_nonlex['dev'])[0])
		# print "x:", np.shape(np.array(data['dev']['x'])[:,0,:,:])
		# print "y:", np.shape(np.array(data['dev']['y']))
		# data['train']['x'] dimensions: (num_batches, a or b index, max_emails, max_email_words, word_vec_dim)
		# Expected input: A emails, B emails, A non-lex, B non-lex
		x_train_lex = get_approach2_lex_input(data, 'train', max_emails)
		x_dev_lex = get_approach2_lex_input(data, 'dev', max_emails)
		x_test_lex = get_approach2_lex_input(data, 'test', max_emails)
		x_train = x_train_lex
		x_train.append(data_nonlex['train'][0])
		x_train.append(data_nonlex['train'][1])
		x_dev = x_dev_lex
		x_dev.append(data_nonlex['dev'][0])
		x_dev.append(data_nonlex['dev'][1])
		x_test = x_test_lex
		x_test.append(data_nonlex['test'][0])
		x_test.append(data_nonlex['test'][1])

	else:
		# Just use lexical features
		# Merge LSTMs
		LSTMs_and_nonlex_ab = LSTMs_a + LSTMs_b
		inputs_and_nonlex_ab = inputs_a + inputs_b
		merged = concatenate(LSTMs_and_nonlex_ab)
		merged_dense = Dense(32, activation='relu')(merged)
	
		# Softmax classification
		dense_out = Dense(1, activation='sigmoid')(merged_dense)
		merged_model = Model(inputs=inputs_and_nonlex_ab, outputs=[dense_out])

		# Prepare input data
		# Expected input: A emails, B emails
		x_train = get_approach2_lex_input(data, 'train', max_emails)
		x_dev = get_approach2_lex_input(data, 'dev', max_emails)
		x_test = get_approach2_lex_input(data, 'test', max_emails)

	merged_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	print("Compiled merged_model!")


	# Fit model
	merged_model.fit(x_train, np.array(data['train']['y']),
		# batch_size={{choice([10, 30, 50])}}, # Tuning
		batch_size=batch_size,
		# epochs={{choice([10, 20, 30])}}, # Tuning
		epochs=num_epochs,
		validation_data=(x_dev, np.array(data['dev']['y'])),
		# callbacks=[metrics]
		)
	print "Fitted merged_model!"

	# Evaluate model
	test_predict = np.asarray(merged_model.predict(x_test)).round()
	test_target = np.array(data['test']['y'])
	test_f1 = f1_score(test_target, test_predict)
	test_recall = recall_score(test_target, test_predict)
	test_precision = precision_score(test_target, test_predict)
	test_acc = float(np.mean(test_predict == test_target))
	print " - test_acc: %f - test_f1: %f - test_precision: %f - test_recall %f" %(test_acc, test_f1, test_precision, test_recall)

	return {'loss': -test_f1, 'status': STATUS_OK, 'model': merged_model}

# AugCNN: Approach 1 ---------------------------------
def get_CNN(num_filters=32, strides=(1,1), activation='relu', max_email_words=50, word_vec_dim=100, isApproach3=False):
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
	# flatten = TimeDistributed(Flatten())(l_pool3)

	# # 2) Simple Conv2D version
	# conv = Conv2D(num_filters, kernel_size=5, strides=1, activation=activation)(cur_input)
	# l_cov1= Conv2D(128, 2, activation='relu', padding='valid')(cur_input)
	# l_pool1 = MaxPool2D(4)(l_cov1)
	# l_cov2 = Conv2D(128, 2, activation='relu', padding='valid')(l_pool1)
	# l_pool2 = MaxPool2D(2)(l_cov2)
	# l_cov3 = Conv2D(128, 2, activation='relu', padding='valid')(l_pool2)
	# l_pool3 = MaxPool2D(2)(l_cov3)  # global max pooling
	# flatten = TimeDistributed(Flatten())(l_pool3)

	# # 2b) Second Conv2D version
	# kernel_size=5
	# conv = Conv2D(num_filters, kernel_size=kernel_size, strides=1, activation=activation)(cur_input)
	# l_cov1= Conv2D(num_filters, kernel_size, activation='relu', padding='valid')(cur_input)
	# l_pool1 = MaxPool2D(kernel_size, padding='valid')(l_cov1)
	# l_cov2 = Conv2D(num_filters, kernel_size, activation='relu', padding='valid')(l_pool1)
	# l_pool2 = MaxPool2D(kernel_size, padding='valid')(l_cov2)
	# # l_cov3 = Conv2D(num_filters, kernel_size, activation='relu', padding='valid')(l_pool2)
	# # l_pool3 = MaxPool2D(35, padding='valid')(l_cov3)  # global max pooling
	# flatten = TimeDistributed(Flatten())(l_pool2)

	# 2c) Third Conv2D version
	# kernel_size=3
	# num_filters=64
	# conv = Conv2D(num_filters, kernel_size=kernel_size, strides=1, activation=activation)(cur_input)
	# l_cov1= Conv2D(num_filters, kernel_size, activation='relu', padding='valid')(cur_input)
	# l_pool1 = MaxPool2D(kernel_size, padding='valid')(l_cov1)
	# l_cov2 = Conv2D(num_filters, kernel_size, activation='relu', padding='valid')(l_pool1)
	# l_pool2 = MaxPool2D(kernel_size, padding='valid')(l_cov2)
	# # l_cov3 = Conv2D(num_filters, kernel_size, activation='relu', padding='valid')(l_pool2)
	# # l_pool3 = MaxPool2D(35, padding='valid')(l_cov3)  # global max pooling
	# flatten = TimeDistributed(Flatten())(l_pool2)

	#2d) Third Conv2D version (simplest)
	# kernel_size=3
	# l_cov1 = Conv2D(64, kernel_size=(kernel_size, word_vec_dim), strides=strides, activation=activation)(cur_input)
	# l_pool1 = MaxPooling2D(pool_size=(word_vec_dim - kernel_size + 1, 1), strides=(2, 2), padding='valid')(l_cov1)
	# # l_cov2 = Conv2D(32, kernel_size=(kernel_size, word_vec_dim), activation='relu', padding='valid')(l_pool1)
	# # l_pool2 = MaxPool2D(pool_size=(word_vec_dim - kernel_size + 1, 1), strides=(2, 2), padding='valid')(l_cov2)
	# flatten = TimeDistributed(Flatten())(l_pool1)

	# 3) Multiple filters version
	# filter_sizes = [3,4,5]
	# conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], word_vec_dim), strides=strides, activation=activation, padding='valid')(cur_input)
	# conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], word_vec_dim), strides=strides, activation=activation, padding='valid')(cur_input)
	# conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], word_vec_dim), strides=strides, activation=activation, padding='valid')(cur_input)
	# maxpool_0 = MaxPool2D(pool_size=(max_email_words - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
	# maxpool_1 = MaxPool2D(pool_size=(max_email_words - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
	# maxpool_2 = MaxPool2D(pool_size=(max_email_words - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)
	# concatenated_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)

	# if isApproach3:
	# 	flatten = TimeDistributed(Flatten())(concatenated_tensor)
	# else:
	# 	flatten = Flatten()(concatenated_tensor)
	# kernel_size=3
	# l_cov1 = Conv2D(64, kernel_size=(kernel_size, word_vec_dim), strides=strides, activation=activation)(cur_input)
	# l_pool1 = MaxPooling2D(pool_size=(word_vec_dim - kernel_size + 1, 1), strides=(2, 2), padding='valid')(l_cov1)
	# # l_cov2 = Conv2D(32, kernel_size=(kernel_size, word_vec_dim), activation='relu', padding='valid')(l_pool1)
	# # l_pool2 = MaxPool2D(pool_size=(word_vec_dim - kernel_size + 1, 1), strides=(2, 2), padding='valid')(l_cov2)
	# flatten = TimeDistributed(Flatten())(l_pool1)

	# 3) Multiple filters version
	filter_sizes = [3,4,5]
	conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], word_vec_dim), strides=strides, activation=activation, padding='valid')(cur_input)
	conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], word_vec_dim), strides=strides, activation=activation, padding='valid')(cur_input)
	conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], word_vec_dim), strides=strides, activation=activation, padding='valid')(cur_input)
	maxpool_0 = MaxPool2D(pool_size=(max_email_words - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
	maxpool_1 = MaxPool2D(pool_size=(max_email_words - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
	maxpool_2 = MaxPool2D(pool_size=(max_email_words - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)
	concatenated_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)

	if isApproach3:
		flatten = TimeDistributed(Flatten())(concatenated_tensor)
	else:
		flatten = Flatten()(concatenated_tensor)

	return cur_input, flatten

def get_nonlex(data):
	# Prepare non-lexical data
	data_nonlex = {}
	splits = ['train', 'dev', 'test']
	for split in splits:
		data_nonlex[split] = []
		data_nonlex[split].append(np.array([np.array(vals)[:,0] for feat, vals in data[split]['non_lex'].iteritems()]).T)
		data_nonlex[split].append(np.array([np.array(vals)[:,1] for feat, vals in data[split]['non_lex'].iteritems()]).T)
		print "shape:", np.shape(data_nonlex[split][0])

	non_lex_dim = len(data['train']['non_lex'])
	input_nonlex_a = Input(shape=(non_lex_dim,), dtype='float32')
	input_nonlex_b = Input(shape=(non_lex_dim,), dtype='float32')
	non_lex_features_a = Dense(input_dim=non_lex_dim,output_dim=non_lex_dim)(input_nonlex_a)
	non_lex_features_b = Dense(input_dim=non_lex_dim,output_dim=non_lex_dim)(input_nonlex_b)
	return data_nonlex, input_nonlex_a, input_nonlex_b, non_lex_features_a, non_lex_features_b

def AugCNN1_full_data():
	pkl_file = 'aug_data/approach1/grouped_100.pkl'
	with open(pkl_file, 'rb') as f:
		data = pickle.load(f)
	num_filters=32
	batch_size=30
	num_epochs=10
	strides=(1, 1)
	activation='relu'
	max_email_words=100
	word_vec_dim=100
	dropout=0.5
	use_non_lex=False
	return data, num_filters, batch_size, num_epochs, strides, activation, max_email_words, word_vec_dim, dropout, use_non_lex

"""
Previous hyperparam tunes:
# dropout_layer = Dropout({{choice([0.2, 0.3, 0.4])}})(merged)
# dropout_layer = Dropout({{uniform(0, 1)}})(merged) 
# merged_model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}}, loss='binary_crossentropy', metrics=['accuracy'])
# epochs={{choice([10, 20, 50])}}
"""
def AugCNN1_full(data, num_filters=32, batch_size=30, num_epochs=10, strides=(1, 1), activation='relu', max_email_words=50, word_vec_dim=100, dropout=0.2, use_non_lex=True):
	
	# Create CNNs
	input_a, CNN_a = get_CNN(num_filters, strides, activation, max_email_words=max_email_words, word_vec_dim=word_vec_dim)
	input_b, CNN_b = get_CNN(num_filters, strides, activation, max_email_words=max_email_words, word_vec_dim=word_vec_dim)

	
	if use_non_lex:
		# Merge non-lexical features
		# Prepare non-lexical data
		data_nonlex, input_nonlex_a, input_nonlex_b, non_lex_features_a, non_lex_features_b = get_nonlex(data)

		# Merge CNNs
		merged = concatenate([CNN_a, non_lex_features_a, CNN_b, non_lex_features_b])
		merged_dense = Dense(32, activation=activation)(merged)
		dropout_layer = Dropout(dropout)(merged_dense)

		# Softmax classification
		dense_out = Dense(1, activation='sigmoid')(dropout_layer)
		merged_model = Model(inputs=[input_a, input_nonlex_a, input_b, input_nonlex_b], outputs=[dense_out])

		# Prepare input data
		print "nl:", np.shape(np.array(data_nonlex['dev'])[0])
		print "x:", np.shape(np.array(data['dev']['x'])[:,0,:,:])
		print "y:", np.shape(np.array(data['dev']['y']))
		x_train = [np.array(data['train']['x'])[:,0,:,:,np.newaxis],
			data_nonlex['train'][0], 
			np.array(data['train']['x'])[:,1,:,:,np.newaxis], 
			data_nonlex['train'][1]]
		x_dev = [np.array(data['dev']['x'])[:,0,:,:,np.newaxis], 
			data_nonlex['dev'][0],
			np.array(data['dev']['x'])[:,1,:,:,np.newaxis], 
			data_nonlex['dev'][1]]
		x_test = [np.array(data['test']['x'])[:,0,:,:,np.newaxis],
			data_nonlex['test'][0],
			np.array(data['test']['x'])[:,1,:,:,np.newaxis],
			data_nonlex['test'][1]]

	else:
		# Just use words
		# Merge CNNs
		merged = concatenate([CNN_a, CNN_b])
		merged_dense = Dense(32, activation=activation)(merged)
		dropout_layer = Dropout(dropout)(merged_dense)
	
		# Softmax classification
		dense_out = Dense(1, activation='sigmoid')(dropout_layer)
		merged_model = Model(inputs=[input_a, input_b], outputs=[dense_out])

		# Prepare input data
		x_train = [np.array(data['train']['x'])[:,0,:,:,np.newaxis],
			np.array(data['train']['x'])[:,1,:,:,np.newaxis]]
		x_dev = [np.array(data['dev']['x'])[:,0,:,:,np.newaxis], 
			np.array(data['dev']['x'])[:,1,:,:,np.newaxis]]
		x_test = [np.array(data['test']['x'])[:,0,:,:,np.newaxis],
			np.array(data['test']['x'])[:,1,:,:,np.newaxis]]

	merged_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	print("Compiled merged_model!")

	# Fit model
	merged_model.fit(x_train, np.array(data['train']['y']),
		batch_size=batch_size,
		epochs=num_epochs,
		validation_data=(x_dev, np.array(data['dev']['y'])),
			#callbacks=[metrics]
			)
	print "Fitted merged_model!"

	# Evaluate model
	test_predict = np.asarray(merged_model.predict(x_test)).round()
	test_target = np.array(data['test']['y'])
	test_f1 = f1_score(test_target, test_predict)
	test_recall = recall_score(test_target, test_predict)
	test_precision = precision_score(test_target, test_predict)
	test_acc = float(np.mean(test_predict == test_target))
	print " - test_acc: %f - test_f1: %f - test_precision: %f - test_recall %f" %(test_acc, test_f1, test_precision, test_recall)

	return {'loss': -test_f1, 'status': STATUS_OK, 'model': merged_model}

# AugCNN: Approach 2 ---------------------------------
def AugCNN2_full_data():
	pkl_file = 'aug_data/approach2/grouped_100.pkl'
	with open(pkl_file, 'rb') as f:
		data = pickle.load(f)
	num_filters=32 # -
	batch_size=30 # -
	num_epochs=10 # -
	strides=(1, 1)
	activation='relu'
	max_email_words=100
	word_vec_dim=100
	dropout=0.5 # -
	use_non_lex=True
	max_emails=5
	return data, num_filters, batch_size, num_epochs, strides, activation, max_email_words, word_vec_dim, dropout, use_non_lex, max_emails

def AugCNN2_full(data, num_filters=32, batch_size=30, num_epochs=10, strides=(1, 1), activation='relu', max_email_words=50, word_vec_dim=100, dropout=0.2, use_non_lex=True, max_emails=5):

	# Get all of person A's emails; get all of B's emails
	# Generate CNN for each email
	# Generate non-lex layer for each email
	# Merge all CNN and non-lex layers
	# Dense layer; softmax
	
	# Create CNNs
	inputs_a = []
	inputs_b = []
	CNNs_a = []
	CNNs_b = []
	for i in range(max_emails):
		# input_a, CNN_a = get_CNN({{choice([16, 24, 32])}}, strides, activation, max_email_words=max_email_words, word_vec_dim=word_vec_dim) # Tuning
		# input_b, CNN_b = get_CNN({{choice([16, 24, 32])}}, strides, activation, max_email_words=max_email_words, word_vec_dim=word_vec_dim) # Tuning
		input_a, CNN_a = get_CNN(num_filters, strides, activation, max_email_words=max_email_words, word_vec_dim=word_vec_dim)
		input_b, CNN_b = get_CNN(num_filters, strides, activation, max_email_words=max_email_words, word_vec_dim=word_vec_dim)
		inputs_a.append(input_a)
		inputs_b.append(input_b)
		CNNs_a.append(CNN_a)
		CNNs_b.append(CNN_b)
	
	if use_non_lex:
		# Get non-lexical features
		data_nonlex, input_nonlex_a, input_nonlex_b, non_lex_features_a, non_lex_features_b = get_nonlex(data)

		# Merge CNNs
		CNNs_and_nonlex_ab = CNNs_a + CNNs_b
		CNNs_and_nonlex_ab.append(non_lex_features_a)
		CNNs_and_nonlex_ab.append(non_lex_features_b)
		inputs_and_nonlex_ab = inputs_a + inputs_b
		inputs_and_nonlex_ab.append(input_nonlex_a)
		inputs_and_nonlex_ab.append(input_nonlex_b)
		merged = concatenate(CNNs_and_nonlex_ab)
		merged_dense = Dense(32, activation=activation, 
			# kernel_regularizer=regularizers.l2(0.01)
		)(merged) # NEW
		# dropout_layer = Dropout({{choice([0.2, 0.3, 0.4])}})(merged_dense) # Tuning
		dropout_layer = Dropout(dropout)(merged_dense)

		# Softmax classification
		dense_out = Dense(1, activation='sigmoid')(dropout_layer)
		merged_model = Model(inputs=inputs_and_nonlex_ab, outputs=[dense_out])

		# Prepare input data
		# print "nl:", np.shape(np.array(data_nonlex['dev'])[0])
		# print "x:", np.shape(np.array(data['dev']['x'])[:,0,:,:])
		# print "y:", np.shape(np.array(data['dev']['y']))
		# data['train']['x'] dimensions: (num_batches, a or b index, max_emails, max_email_words, word_vec_dim, 1)
		# Expected input: A emails, B emails, A non-lex, B non-lex
		x_train_lex = get_approach2_lex_input(data, 'train', max_emails, isCNN=True)
		x_dev_lex = get_approach2_lex_input(data, 'dev', max_emails, isCNN=True)
		x_test_lex = get_approach2_lex_input(data, 'test', max_emails, isCNN=True)

		# x_train = x_train_lex
		# x_train.append(data_nonlex['train'][0])
		# x_train.append(data_nonlex['train'][1])
		x_train = x_train_lex
		x_train.append(data_nonlex['train'][0])
		x_train.append(data_nonlex['train'][1])
		x_dev = x_dev_lex
		x_dev.append(data_nonlex['dev'][0])
		x_dev.append(data_nonlex['dev'][1])
		x_test = x_test_lex
		x_test.append(data_nonlex['test'][0])
		x_test.append(data_nonlex['test'][1])

	else:
		# Just use words
		# Merge CNNs
		CNNs_ab = CNNs_a + CNNs_b
		inputs_ab = inputs_a + inputs_b
		merged = concatenate(CNNs_ab)
		merged_dense = Dense(32, activation=activation)(merged)
		# dropout_layer = Dropout({{choice([0.1, 0.2, 0.3])}})(merged_dense)
		dropout_layer = Dropout(dropout)(merged_dense)
	
		# Softmax classification
		dense_out = Dense(1, activation='sigmoid')(dropout_layer)
		merged_model = Model(inputs=inputs_ab, outputs=[dense_out])

		# Prepare input data
		# Expected input: A emails, B emails
		x_train = get_approach2_lex_input(data, 'train', max_emails, isCNN=True)
		x_dev = get_approach2_lex_input(data, 'dev', max_emails, isCNN=True)
		x_test = get_approach2_lex_input(data, 'test', max_emails, isCNN=True)

	merged_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	print("Compiled merged_model!")

	# Fit model
	merged_model.fit(x_train, np.array(data['train']['y']),
		# batch_size={{choice([10, 30, 50])}}, # Tuning
		batch_size=batch_size,
		# epochs={{choice([50, 75, 100])}}, # Tuning
		epochs=num_epochs,
		validation_data=(x_dev, np.array(data['dev']['y'])),
			#callbacks=[metrics]
			)
	print "Fitted merged_model!"

	# Evaluate model
	test_predict = np.asarray(merged_model.predict(x_test)).round()
	test_target = np.array(data['test']['y'])
	test_f1 = f1_score(test_target, test_predict)
	test_recall = recall_score(test_target, test_predict)
	test_precision = precision_score(test_target, test_predict)
	test_acc = float(np.mean(test_predict == test_target))
	print " - test_acc: %f - test_f1: %f - test_precision: %f - test_recall %f" %(test_acc, test_f1, test_precision, test_recall)

	return {'loss': -test_f1, 'status': STATUS_OK, 'model': merged_model}


# AugCNNLSTM: Approach 3 ---------------------------------
def AugCNNLSTM3_full_data():
	pkl_file = 'aug_data/approach2/grouped_100.pkl'
	with open(pkl_file, 'rb') as f:
		data = pickle.load(f)
	num_filters=32
	batch_size=30 # -
	num_epochs=10 # -
	strides=(1, 1)
	activation='relu'
	max_email_words=100
	word_vec_dim=100
	dropout=0.5
	use_non_lex=True
	max_emails=5
	output_dim=100 # -
	return data, num_filters, batch_size, num_epochs, strides, activation, max_email_words, word_vec_dim, dropout, use_non_lex, max_emails, output_dim

def AugCNNLSTM3_full(data, num_filters=32, batch_size=30, num_epochs=10, strides=(1, 1), activation='relu', max_email_words=50, word_vec_dim=100, dropout=0.2, use_non_lex=True, max_emails=5, output_dim=100):
	# Get all of person A's emails; get all of B's emails
	# Generate CNN for each email
	# Feed CNN outputs to LSTM for person A, person B
	# Generate non-lex layer for each email
	# Merge all LSTM and non-lex layers
	# Dense layer; softmax
	
	# Create CNNs
	inputs_a = []
	inputs_b = []
	CNNs_a = []
	CNNs_b = []
	for i in range(max_emails):
		input_a, CNN_a = get_CNN(num_filters, strides, activation, max_email_words=max_email_words, word_vec_dim=word_vec_dim, isApproach3=True)
		input_b, CNN_b = get_CNN(num_filters, strides, activation, max_email_words=max_email_words, word_vec_dim=word_vec_dim, isApproach3=True)
		inputs_a.append(input_a)
		inputs_b.append(input_b)
		CNNs_a.append(CNN_a)
		CNNs_b.append(CNN_b)

	# Merge CNNs
	merged_CNNs_a = concatenate(CNNs_a)
	merged_CNNs_b = concatenate(CNNs_b)

	# Feed CNNs to LSTM
	# TODO(michelle): look into dims below!!
	# LSTM_a_cnns = TimeDistributed(lstm_model)(merged_CNNs_a)
	# LSTM_b_cnns = TimeDistributed(lstm_model)(merged_CNNs_b)
	# merged_CNNs_a = TimeDistributed()(concat_CNNs_a)
	# merged_CNNs_b = TimeDistributed()(concat_CNNs_b)

	# LSTM_a = LSTM({{choice([80, 100, 120])}}, dropout=dropout)(merged_CNNs_a) # Tuning
	# LSTM_b = LSTM({{choice([80, 100, 120])}}, dropout=dropout)(merged_CNNs_b) # Tuning
	LSTM_a = LSTM(output_dim, dropout=dropout)(merged_CNNs_a)
	LSTM_b = LSTM(output_dim, dropout=dropout)(merged_CNNs_b)

	if use_non_lex:
		# Get non-lexical features
		data_nonlex, input_nonlex_a, input_nonlex_b, non_lex_features_a, non_lex_features_b = get_nonlex(data)

		# Merge LSTM with non-lex
		LSTMs_and_nonlex_ab = [LSTM_a, LSTM_b, non_lex_features_a, non_lex_features_b]

		# TODO: check!
		inputs_and_nonlex_ab = inputs_a + inputs_b
		inputs_and_nonlex_ab.append(input_nonlex_a)
		inputs_and_nonlex_ab.append(input_nonlex_b)

		merged_LSTMs = concatenate(LSTMs_and_nonlex_ab)
		merged_dense = Dense(32, activation=activation,
			# kernel_regularizer=regularizers.l2(0.01)
			)(merged_LSTMs) # TUNING!
		dropout_layer = Dropout(dropout)(merged_dense)

		# Softmax classification
		dense_out = Dense(1, activation='sigmoid')(dropout_layer)
		merged_model = Model(inputs=inputs_and_nonlex_ab, outputs=[dense_out])

		# Prepare input data
		# data['train']['x'] dimensions: (num_batches, a or b index, max_emails, max_email_words, word_vec_dim, 1)
		# Expected input: A emails, B emails, A non-lex, B non-lex
		x_train_lex = get_approach2_lex_input(data, 'train', max_emails, isCNN=True)
		x_dev_lex = get_approach2_lex_input(data, 'dev', max_emails, isCNN=True)
		x_test_lex = get_approach2_lex_input(data, 'test', max_emails, isCNN=True)

		x_train = x_train_lex
		x_train.append(data_nonlex['train'][0])
		x_train.append(data_nonlex['train'][1])
		x_dev = x_dev_lex
		x_dev.append(data_nonlex['dev'][0])
		x_dev.append(data_nonlex['dev'][1])
		x_test = x_test_lex
		x_test.append(data_nonlex['test'][0])
		x_test.append(data_nonlex['test'][1])

	else:
		# Just use words
		# Merge LSTM with non-lex
		LSTMs_ab = [LSTM_a, LSTM_b]

		# TODO: check!
		inputs_ab = inputs_a + inputs_b

		merged_LSTMs = concatenate(LSTMs_ab)
		merged_dense = Dense(32, activation=activation)(merged_LSTMs)
		dropout_layer = Dropout(dropout)(merged_dense)
	
		# Softmax classification
		dense_out = Dense(1, activation='sigmoid')(dropout_layer)
		merged_model = Model(inputs=inputs_ab, outputs=[dense_out])

		# Prepare input data
		# Expected input: A emails, B emails
		x_train = get_approach2_lex_input(data, 'train', max_emails, isCNN=True)
		x_dev = get_approach2_lex_input(data, 'dev', max_emails, isCNN=True)
		x_test = get_approach2_lex_input(data, 'test', max_emails, isCNN=True)

	merged_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	# merged_model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
	# merged_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy']) # Testing

	print("Compiled merged_model!")
	# merged_model.summary() # TEMP

	# Fit model
	merged_model.fit(x_train, np.array(data['train']['y']),
		# batch_size={{choice([15, 30, 45])}}, # Tuning
		batch_size=batch_size,
		# epochs={{choice([30, 40, 50, 60])}}, # Tuning
		epochs=num_epochs,
		validation_data=(x_dev, np.array(data['dev']['y'])),
			#callbacks=[metrics]
			)
	print "Fitted merged_model!"

	# Evaluate model
	test_predict = np.asarray(merged_model.predict(x_test)).round()
	test_target = np.array(data['test']['y'])
	test_f1 = f1_score(test_target, test_predict)
	test_recall = recall_score(test_target, test_predict)
	test_precision = precision_score(test_target, test_predict)
	test_acc = float(np.mean(test_predict == test_target))
	print " - test_acc: %f - test_f1: %f - test_precision: %f - test_recall %f" %(test_acc, test_f1, test_precision, test_recall)

	return {'loss': -test_f1, 'status': STATUS_OK, 'model': merged_model}



# Data processing ---------------------------------
"""
Vectorizes email text data
- TODO: decide best way to capture email in terms of word vecs:
	- [currently] Sequence of word vectors (capped at max_email_words)
	OR
	- Sum of sentence vectors (each sentence capped at max_sentence_words) (what we did before)
	- Paragraph vectors
"""
def get_vectorized_email(email, wordVectors, max_email_words=100, word_vec_dim=100):
	result = np.zeros((max_email_words, word_vec_dim))
	if email == "NO_EMAILS":
		return result
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
def processData1(raw_xa_file, raw_xb_file, raw_y_file, non_lex_feats_files, pkl_file):
	data = {}
	data['x'] = []
	data['y'] = []
	data['non_lex'] = {}

	# Load word vectors and tokens
	with open('aug_data/glove.6B.100d.txt', 'r') as f:
		wordVectors = {}
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			wordVectors[word] = coefs
	print("Loaded GloVe!")

	# Load npy file
	raw_xa = np.load(raw_xa_file)
	raw_xb = np.load(raw_xb_file)
	raw_y = np.load(raw_y_file)
	raw_non_lex_feats = {}
	for file_prefix in non_lex_feats_files:
		feat_name = file_prefix.split('/')[-1] # last part of filename
		raw_non_lex_feats[feat_name] = [
			np.load(file_prefix + '_1.npy'),
			np.load(file_prefix + '_2.npy')
		]
		data['non_lex'][feat_name] = []

	# Separate A's and B's emails for each pairing
	n_pairs = len(raw_xa)

	for pair_i in range(n_pairs):
		a_email = get_vectorized_email(raw_xa[pair_i], wordVectors, max_email_words=150)
		b_email = get_vectorized_email(raw_xb[pair_i], wordVectors, max_email_words=150)
		data['x'].append([a_email, b_email])
		data['y'].append([raw_y[pair_i]])
		for feat_name in raw_non_lex_feats:
			data['non_lex'][feat_name].append(
				[raw_non_lex_feats[feat_name][0][pair_i],
				raw_non_lex_feats[feat_name][1][pair_i]])
	print("Read in data!")

	# Shuffle data
	np.random.seed(10)
	shuffle_indices = np.random.permutation(np.arange(n_pairs))
	data['x'] = [data['x'][i] for i in shuffle_indices]
	data['y'] = [data['y'][i] for i in shuffle_indices]
	for feat_name in raw_non_lex_feats:
		data['non_lex'][feat_name] = [data['non_lex'][feat_name][i] for i in shuffle_indices]
	print("Shuffled data!")

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
	# X input
	split_data['train']['x'] = [data['x'][i] for i in range(train_cutoff)]
	split_data['dev']['x'] = [data['x'][i] for i in range(train_cutoff, dev_cutoff)]
	split_data['test']['x'] = [data['x'][i] for i in range(dev_cutoff, test_cutoff)]
	# Y input
	split_data['train']['y'] = [data['y'][i] for i in range(train_cutoff)]
	split_data['dev']['y'] = [data['y'][i] for i in range(train_cutoff, dev_cutoff)]
	split_data['test']['y'] = [data['y'][i] for i in range(dev_cutoff, test_cutoff)]
	# Non-lex input
	split_data['train']['non_lex'] = {}
	split_data['dev']['non_lex'] = {}
	split_data['test']['non_lex'] = {}
	for feat_name in raw_non_lex_feats:
		split_data['train']['non_lex'][feat_name] = [data['non_lex'][feat_name][i] for i in range(train_cutoff)]
		split_data['dev']['non_lex'][feat_name] = [data['non_lex'][feat_name][i] for i in range(train_cutoff, dev_cutoff)]
		split_data['test']['non_lex'][feat_name] = [data['non_lex'][feat_name][i] for i in range(dev_cutoff, test_cutoff)]

	print("Split data!")	

	# Save shuffled, split data to pickle
	with open(pkl_file, 'wb') as f:
		# print("split_data:", split_data)
		pickle.dump(split_data, f)
	print("Saved data!")

	return split_data

def processData2(raw_xa_file, raw_xb_file, raw_y_file, non_lex_feats_files, pkl_file):
	max_num_emails = 5
	max_email_words = 20
	word_vec_dim = 100

	data = {}
	data['x'] = []
	data['y'] = []
	data['non_lex'] = {}

	# Load word vectors and tokens
	with open('aug_data/glove.6B.100d.txt', 'r') as f:
		wordVectors = {}
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			wordVectors[word] = coefs
	print("Loaded GloVe!")

	# Load npy file
	raw_xa = np.load(raw_xa_file)
	raw_xb = np.load(raw_xb_file)
	raw_y = np.load(raw_y_file)
	# raw_non_lex_feats = {}
	# for file_prefix in non_lex_feats_files:
	# 	feat_name = file_prefix.split('/')[-1] # last part of filename
	# 	raw_non_lex_feats[feat_name] = [
	# 		np.load(file_prefix + '_1.npy'),
	# 		np.load(file_prefix + '_2.npy')
	# 	]
	# 	data['non_lex'][feat_name] = []

	# Separate A's and B's emails for each pairing
	lines_in_xa = len(raw_xa)
	lines_in_xb = len(raw_xb)
	n_pairs = 0

	a_emails_aggregated = []
	a_emails = np.zeros(max_num_emails + 1)
	for line_number in range(lines_in_xa):
		if raw_xa[line_number] == "---":
			# Keep track of number of pairs
			n_pairs += 1
			if a_emails.shape[0] < max_num_emails:
				result = np.zeros((max_num_emails, max_email_words, word_vec_dim))
				result[:a_emails.shape[0], :a_emails.shape[1], :a_emails.shape[2]] = a_emails
				a_emails_aggregated.append(result)
			else:
				a_emails_aggregated.append(a_emails)
			#print(a_emails.shape)
			a_emails = np.zeros(max_num_emails + 1)
		else:
			if a_emails.shape[0] == max_num_emails:
				continue
			if not np.array_equal(a_emails, np.zeros(max_num_emails + 1)):
				#print(a_emails.shape)
				a_email = np.expand_dims(get_vectorized_email(raw_xa[line_number], wordVectors, max_email_words, word_vec_dim), axis=0)
				#print(a_email.shape)
				a_emails = np.vstack((a_emails, a_email))
			else:
				email_vec = np.expand_dims(get_vectorized_email(raw_xa[line_number], wordVectors, max_email_words, word_vec_dim), axis=0)
				a_emails = email_vec

	b_emails_aggregated = []
	b_emails = np.zeros(max_num_emails + 1)
	n_pairs = 0
	for line_number in range(lines_in_xb):
		if raw_xb[line_number] == "---":
			n_pairs += 1
			if b_emails.shape[0] < max_num_emails:
				result = np.zeros((max_num_emails, max_email_words, word_vec_dim))
				result[:b_emails.shape[0], :b_emails.shape[1]] = b_emails
				b_emails_aggregated.append(result)
			else:
				b_emails_aggregated.append(b_emails)
			b_emails = np.zeros(max_num_emails + 1)
		else:
			# if there are no corresponding emails
			if raw_xb[line_number] == "NO_EMAILS":
				b_emails = np.zeros((max_num_emails, max_email_words, word_vec_dim))
				continue
			if b_emails.shape[0] == max_num_emails:
				continue
			if not np.array_equal(b_emails, np.zeros(max_num_emails + 1)):
				b_email = np.expand_dims(get_vectorized_email(raw_xb[line_number], wordVectors, max_email_words, word_vec_dim), axis=0)
				b_emails = np.vstack((b_emails, b_email))
			else:
				email_vec = np.expand_dims(get_vectorized_email(raw_xb[line_number], wordVectors, max_email_words, word_vec_dim), axis=0)
				b_emails = email_vec

	for line_number in range(n_pairs):
		data['x'].append([a_emails_aggregated[line_number], b_emails_aggregated[line_number]])
		data['y'].append([raw_y[line_number]])
		# for feat_name in raw_non_lex_feats:
		# 	data['non_lex'][feat_name].append(
		# 		[raw_non_lex_feats[feat_name][0][line_number],
		# 		raw_non_lex_feats[feat_name][1][line_number]])

	print("Read in data!")

	# Shuffle data
	np.random.seed(10)
	shuffle_indices = np.random.permutation(np.arange(n_pairs))
	data['x'] = [data['x'][i] for i in shuffle_indices]
	data['y'] = [data['y'][i] for i in shuffle_indices]
	# for feat_name in raw_non_lex_feats:
	# 	data['non_lex'][feat_name] = [data['non_lex'][feat_name][i] for i in shuffle_indices]
	print("Shuffled data!")

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
	# X input
	split_data['train']['x'] = [data['x'][i] for i in range(train_cutoff)]
	split_data['dev']['x'] = [data['x'][i] for i in range(train_cutoff, dev_cutoff)]
	split_data['test']['x'] = [data['x'][i] for i in range(dev_cutoff, test_cutoff)]
	# Y input
	split_data['train']['y'] = [data['y'][i] for i in range(train_cutoff)]
	split_data['dev']['y'] = [data['y'][i] for i in range(train_cutoff, dev_cutoff)]
	split_data['test']['y'] = [data['y'][i] for i in range(dev_cutoff, test_cutoff)]
	# Non-lex input
	split_data['train']['non_lex'] = {}
	split_data['dev']['non_lex'] = {}
	# split_data['test']['non_lex'] = {}
	# for feat_name in raw_non_lex_feats:
	# 	split_data['train']['non_lex'][feat_name] = [data['non_lex'][feat_name][i] for i in range(train_cutoff)]
	# 	split_data['dev']['non_lex'][feat_name] = [data['non_lex'][feat_name][i] for i in range(train_cutoff, dev_cutoff)]
	# 	split_data['test']['non_lex'][feat_name] = [data['non_lex'][feat_name][i] for i in range(dev_cutoff, test_cutoff)]

	print "split_data['train']['x'] shape:", np.shape(np.array(split_data['train']['x']))
	print("Split data!")

	# Save shuffled, split data to pickle
	with open(pkl_file, 'wb') as f:
		# print("split_data:", split_data)
		pickle.dump(split_data, f)
	print("Saved data!")

	#print(np.array(split_data['train']['x']).shape)
	return split_data

def main(args):
	# TODO: update to true data file
	if args.thread and args.approach == 1:
		if args.fullEmailsThread:
			raw_xa_file = "aug_data/approach1/thread_content_1_extended.npy"
			raw_xb_file = "aug_data/approach1/thread_content_2_extended.npy"
			raw_y_file = "aug_data/approach1/thread_labels_approach_1_extended.npy"
			pkl_file = 'aug_data/approach1/thread_extended_150.pkl'
		else:
			raw_xa_file = 'aug_data/approach1/thread_content_1.npy'
			raw_xb_file = 'aug_data/approach1/thread_content_2.npy'
			raw_y_file = 'aug_data/approach1/thread_labels_approach_1.npy'
			pkl_file = 'aug_data/approach1/thread.pkl'
	elif not args.thread and args.approach == 1:
		if args.fullEmailsGrouped:
			raw_xa_file = 'aug_data/approach1/email_contents_grouped_1_extended.npy'
			raw_xb_file = 'aug_data/approach1/email_contents_grouped_2_extended.npy'
			raw_y_file = 'aug_data/approach1/labels_grouped_approach_1_extended.npy'
			pkl_file = 'aug_data/approach1/grouped_150.pkl' # max_email_words=150
		else: 
			raw_xa_file = 'aug_data/approach1/email_contents_grouped_1.npy'
			raw_xb_file = 'aug_data/approach1/email_contents_grouped_2.npy'
			raw_y_file = 'aug_data/approach1/labels_grouped.npy'
			# pkl_file = 'aug_data/approach1/grouped.pkl' # max_email_words=50
			# pkl_file = 'aug_data/approach1/grouped_100.pkl' # max_email_words=100
			pkl_file = 'aug_data/approach1/grouped_200.pkl' # max_email_words=200

	elif not args.thread and (args.approach == 2 or args.approach == 3):
		if args.fullEmailsGrouped:
			raw_xa_file = 'aug_data/approach2/email_contents_grouped_1_individual_extended.npy'
			raw_xb_file = 'aug_data/approach2/email_contents_grouped_2_individual_extended.npy'
			raw_y_file = 'aug_data/approach2/labels_grouped_approach_1_extended.npy'
			pkl_file = 'aug_data/approach2/grouped_150_extended.pkl'
		else: 
			raw_xa_file = 'aug_data/approach2/email_contents_grouped_1_individual.npy'
			raw_xb_file = 'aug_data/approach2/email_contents_grouped_2_individual.npy'
			raw_y_file = 'aug_data/approach2/labels_grouped_approach_1.npy'
			pkl_file = 'aug_data/approach2/grouped_100.pkl'
	elif args.thread and (args.approach == 2 or args.approach == 3):
		if args.fullEmailsThread:
			raw_xa_file = 'aug_data/approach2/thread_content_1_individual_extended.npy'
			raw_xb_file = 'aug_data/approach2/thread_content_2_individual_extended.npy'
			raw_y_file = 'aug_data/approach2/thread_labels_extended.npy'
			pkl_file = 'aug_data/approach2/thread_labels_20_extended.pkl'
		else:
			raw_xa_file = 'aug_data/approach2/thread_content_1_individual.npy'
			raw_xb_file = 'aug_data/approach2/thread_content_2_individual.npy'
			raw_y_file = 'aug_data/approach2/thread_labels_approach_1.npy'
			pkl_file = 'aug_data/approach2/thread_labels_100.pkl'

	# Non-lexical feature file names
	# (excluding "_1.npy" or "_2.npy" portion)
	if args.fullEmailsGrouped:
		non_lex_feats_files = ['aug_data/approach1/non_lex_feats/avg_num_recipients_extended', 'aug_data/approach1/non_lex_feats/avg_num_tokens_per_email_extended']
	else:
		non_lex_feats_files = ['aug_data/approach1/non_lex_feats/avg_num_recipients', 'aug_data/approach1/non_lex_feats/avg_num_tokens_per_email']

	# Prepare train/dev/test data
	if args.prepareData:
		# Approach 1 data
		if args.approach == 1:
			print("Preparing Approach 1 data!")
			data = processData1(raw_xa_file, raw_xb_file, raw_y_file, non_lex_feats_files, pkl_file)

		# Approach 2 and 3 data
		elif args.approach == 2 or args.approach == 3:
			print("Preparing Approach ", args.approach, " data!")
			data = processData2(raw_xa_file, raw_xb_file, raw_y_file, non_lex_feats_files, pkl_file)

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
			# Approach 1 LSTM
			if args.model == 'LSTM':
				print("Running Approach 1 LSTM!")
				# print("trainX shape:", np.shape(data['train']['x']))
				# print("trainY shape:", np.shape(data['train']['y']))
				# print("devX shape:", np.shape(data['dev']['x']))
				# print("sliced trainX shape:", np.shape(np.array(data['train']['x'])[:,0,:,:]))
				# TEMP
				print "non_lex train shape:", len(data['train']['non_lex'])
				print "non_lex dev shape:", np.shape(np.array(data['dev']['non_lex']['avg_num_tokens_per_email']))
				print "non_lex test shape:", np.shape(np.array(data['test']['non_lex']['avg_num_tokens_per_email']))
				if args.tuneParams:
					# Hyperparameter tuning
					functions=[get_nonlex]
					best_run, best_model = optim.minimize(
						model=AugLSTM1_full,
						data=AugLSTM1_full_data,
						functions=functions,
						algo=rand.suggest,
						max_evals=5,
						trials=Trials())
					print("best_run:", best_run)
				else:
					AugLSTM1_full(data,
						output_dim=100,
						dropout=0.3,
						batch_size=30,
						num_epochs=50, # 50
						max_email_words=100,
						word_vec_dim=100,
						use_non_lex=args.useNonLex)

			# Approach 1 CNN
			elif args.model == 'CNN':
				print("Running Approach 1 CNN!")
				if args.tuneParams:
					# Hyperparameter tuning
					functions=[get_CNN, get_nonlex]
					best_run, best_model = optim.minimize(
						model=AugCNN1_full,
						data=AugCNN1_full_data,
						functions=functions,
						algo=rand.suggest,
						max_evals=5,
						trials=Trials())
					print("best_run:", best_run)

					# # Show results of best model
					# if args.useNonLex:
					# 	x_test = [np.array(data['test']['x'])[:,0,:,:,np.newaxis], data_nonlex['test'][0], np.array(data['test']['x'])[:,1,:,:,np.newaxis], data_nonlex['test'][1]]
					# else:
					# 	x_test = [np.array(data['test']['x'])[:,0,:,:,np.newaxis], np.array(data['test']['x'])[:,1,:,:,np.newaxis]]
					# y_test = data['test']['y']
					# best_model.evaluate(x_test, y_test)
				else:
					AugCNN1_full(data,
						num_filters=32, # 32
						batch_size=30,
						num_epochs=90,
						strides=(1, 1),
						activation='relu',
						max_email_words=150,
						word_vec_dim=100,
						dropout=0.2,
						use_non_lex=args.useNonLex)

		# Approach 2 model
		elif args.approach == 2:
			# Approach 2 LSTM
			if args.model == 'LSTM':
				print("Running Approach 2 LSTM!")
				print("trainX shape:", np.shape(data['train']['x']))
				if args.tuneParams:
					# Hyperparameter tuning
					functions=[get_LSTM, get_nonlex, get_approach2_lex_input]
					best_run, best_model = optim.minimize(
						model=AugLSTM2_full,
						data=AugLSTM2_full_data,
						functions=functions,
						algo=rand.suggest,
						max_evals=5,
						trials=Trials())
					print("best_run:", best_run)
				else:
					AugLSTM2_full(data,
						output_dim=100,
						dropout=0.3,
						batch_size=30,
						num_epochs=100, # 50
						max_email_words=100,
						word_vec_dim=100,
						use_non_lex=args.useNonLex)
			
			# Approach 2 CNN
			elif args.model == 'CNN':
				print("Running Approach 2 CNN!")
				if args.tuneParams:
					# Hyperparameter tuning
					functions=[get_CNN, get_nonlex, get_approach2_lex_input]
					best_run, best_model = optim.minimize(
						model=AugCNN2_full,
						data=AugCNN2_full_data,
						functions=functions,
						algo=rand.suggest,
						max_evals=5,
						trials=Trials())
					print("best_run:", best_run)
				else:
					AugCNN2_full(data,
						num_filters=32, # 32
						batch_size=10, # 30
						num_epochs=10,
						strides=(1, 1),
						activation='relu',
						max_email_words=20,
						word_vec_dim=100,
						dropout=0.2,
						use_non_lex=args.useNonLex)

		# Approach 3 model
		elif args.approach == 3:
			# CNN-LSTM model
			print("Running Approach 3 CNN-LSTM!")
			print("trainX shape:", np.shape(data['train']['x']))
			if args.tuneParams:
				# Hyperparameter tuning
					functions=[get_CNN, get_nonlex, get_approach2_lex_input]
					best_run, best_model = optim.minimize(
						model=AugCNNLSTM3_full,
						data=AugCNNLSTM3_full_data,
						functions=functions,
						algo=rand.suggest,
						max_evals=5,
						trials=Trials())
					print("best_run:", best_run)
			else:
				AugCNNLSTM3_full(data,
						num_filters=32,
						batch_size=60, # 30
						output_dim=100, # 100
						num_epochs= 35, #40, # 90
						strides=(1, 1),
						activation='relu',
						max_email_words=20,
						word_vec_dim=100,
						dropout=0.05,
						use_non_lex=args.useNonLex)


if __name__ == "__main__":
	# Prepare command line flags
	parser = argparse.ArgumentParser(description='Run augmented LSTM or CNN models')
	parser.add_argument('--prepareData', type=bool, default=False, help='whether to prepare data for the model (default=False)')
	parser.add_argument('--approach', type=int, default=1, help="which number approach to use (default=1)")
	parser.add_argument('--model', type=str, default='LSTM', help="which model to use (default=LSTM)")
	parser.add_argument('--tuneParams', type=bool, default=False, help="whether to tune hyperparams with hyperas (default=False)")
	parser.add_argument('--thread', type=bool, default=False, help="whether to use thread data or grouped data")
	parser.add_argument('--useNonLex', type=bool, default=False, help="whether to use non-lexical (structural) features (default=False)")
	parser.add_argument('--fullEmailsThread', type=bool, default=False, help="whether to use full thread data or regular thread data")
	parser.add_argument('--fullEmailsGrouped', type=bool, default=False, help="whether to use full group data or regular thread data")

	args = parser.parse_args()
	main(args)
