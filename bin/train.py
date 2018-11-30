# -*- coding: utf-8 -*-
# @Author: rlin
# @Date:   2018-07-31 09:10:18
# @Last Modified by:   rlin
# @Last Modified time: 2018-08-09 16:36:15

import logging

from os import listdir

import numpy as np
import pandas as pd

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from tools import preprocess_data

import lib

# Classification Parameters
MAX_LABEL = 7 # Number of classes to predict. Update if number of case types changes.

MAX_WORDS = 500 # maximum number of words to keep, only most common MAX_WORDS will be kept

# Model Training Parameters
NUM_EPOCHS = 25 # The number of iterations we run through the entire dataset. Each epoch will affect the model weights.
BATCH_SIZE = 256 # start high

def load_data():
	logging.info('Loading train data')

	# Get data directory
	train_data_directory = lib.get_conf('raw_labeled_data_directory')

	# Get file name
	train_data_file_name = listdir(train_data_directory)[0]

	# Generate complete file path
	train_data_path = train_data_directory + '\\' + train_data_file_name

	pd_data = pd.read_csv(train_data_path)

	return pd_data

def build_keras_lstm_model():
	'''
	Defines Neural Network Architecture
	'''
	# Instantiate Keras Model Sequence
	model = Sequential()

	# Input Layer
	model.add(LSTM(128, return_sequences=True, input_shape=(None, MAX_WORDS)))
	model.add(Dropout(0.5))
	model.add(LSTM(100, input_shape=(None, MAX_WORDS)))
	model.add(Dropout(0.5))

	model.add(Dense(7, activation='softmax')) # Output layer

	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

	return model


def train_model(x_train, y_train):
	model_dir_path = lib.get_conf('model_directory')

	logging.info('Initializing model')

	classifier = build_keras_lstm_model()

	logging.info('Training model')
	classifier.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, shuffle=True)

	model_name = 'keras_lstm_epochs{num_epochs}_batchsize{batch_size}_words{max_words}.h5'.format(
		num_epochs=NUM_EPOCHS, 
		batch_size=BATCH_SIZE,
		max_words=MAX_WORDS)

	model_output_path = model_dir_path + '\\' + model_name

	logging.info('Saving trained model')
	classifier.save(model_output_path)


def main():
	# Initialize logger
	LOG_FILENAME = 'case_automation.log'
	logging.basicConfig(filename=LOG_FILENAME,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

	# Load data
	pd_data = load_data()

	# Process data for training
	x_train, y_train = preprocess_data.process_data(pd_data, train=True)

	# Trains and saves model to model_directory specified in config.yaml
	train_model(x_train, y_train)

	pass

if __name__ == '__main__':
    main()