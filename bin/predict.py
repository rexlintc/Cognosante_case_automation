# -*- coding: utf-8 -*-
# @Author: rlin
# @Date:   2018-07-31 09:10:41
# @Last Modified by:   rlin
# @Last Modified time: 2018-08-09 16:03:52

import logging

import numpy as np
import pandas as pd

from keras.models import load_model

from tools import preprocess_data
import lib

from os import listdir

n_words = 0 # placeholder for number of actual words collected from Tokenizer
EMBEDDING_SIZE = 50
MAX_LABEL = 7
WORDS_FEATURE = 'words'  # Name of the input words feature.


def load_data():
	input_path = lib.get_conf('unclassified_data_directory')

	pd_data = pd.read_csv(input_path)

	return pd_data

def predict(x_pred):
	# Predict.
	model_dir_path = lib.get_conf('model_directory')

	# Load saved model
	model_name = listdir(model_dir_path)[-1]

	model_path = model_dir_path +'\\' + model_name

	logging.info('Loading saved model')
	classifier = load_model(model_path)

	predictions = classifier.predict_classes(x_pred)

	# To output probability of classes
	#prediction_probability = classifer.predict(x_pred)

	output_dir = lib.get_conf('prediction_output_directory')
	output_path = output_dir + '\\tf_prediction_output.csv'

	pd_prediction_output = pd.DataFrame(predictions)

	logging.info('Writing prediction output to {}'.format(output_dir))
	pd_prediction_output.to_csv(output_path)


def main():
	# Initialize logger
	LOG_FILENAME = 'case_automation.log'
	logging.basicConfig(filename=LOG_FILENAME,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

	logging.info('Loading unclassified data for prediction')
	pd_data = load_data()

	logging.info('Processing data')
	x_pred = preprocess_data.process_data(pd_data)

	predict(x_pred)

	pass

if __name__ == '__main__':
    main()