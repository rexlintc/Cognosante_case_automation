# -*- coding: utf-8 -*-
# @Author: rlin
# @Date:   2018-07-31 09:11:06
# @Last Modified by:   rlin
# @Last Modified time: 2018-08-09 10:50:34

import logging

import numpy as np
import pandas as pd

from sklearn import metrics

from keras.models import load_model

from tools import preprocess_data
import lib

from os import listdir


def evaluate(x_test, y_test):
	model_dir_path = lib.get_conf('model_directory')

	# Load saved model
	model_name = listdir(model_dir_path)[-1]

	model_path = model_dir_path +'\\' + model_name

	classifier = load_model(model_path)

	y_pred = classifier.predict_classes(x_test)

	# Instantiate empty output dataframe
	pd_eval_output = pd.DataFrame()
	pd_eval_output['Actual'] = pd.Series(y_test)
	pd_eval_output['Predicted'] = pd.Series(y_pred)

	eval_output_directory = lib.get_conf('prediction_output_directory')
	eval_output_file_name = 'tf_evaluation_output.csv'
	eval_output_path = eval_output_directory + '\\' + eval_output_file_name

	# Write out evaluation output
	pd_eval_output.to_csv(eval_output_path)

	score = metrics.accuracy_score(y_test, y_pred)

	cm = metrics.confusion_matrix(y_test, y_pred)

	print('Model Accuracy: {}'.format(score)) #For Demo purposes
	logging.info('Model Accuracy: {}'.format(score))


def load_data():
	logging.info('Loading evaluation data')
	eval_data_directory = lib.get_conf('raw_labeled_data_directory')

	eval_data_file_name = listdir(eval_data_directory)[0]

	eval_data_path = eval_data_directory + '\\' + eval_data_file_name

	pd_eval_data = pd.read_csv(eval_data_path)

	return pd_eval_data

def main():
	# Initialize logger
	LOG_FILENAME = 'case_automation.log'
	logging.basicConfig(filename=LOG_FILENAME,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

	pd_data = load_data()

	x_test, y_test = preprocess_data.process_data(pd_data, evaluate=True)

	evaluate(x_test, y_test)

	pass

if __name__ == '__main__':
    main()