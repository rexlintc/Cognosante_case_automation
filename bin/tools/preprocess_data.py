# -*- coding: utf-8 -*-
# @Author: rlin
# @Date:   2018-07-31 16:17:05
# @Last Modified by:   rlin
# @Last Modified time: 2018-08-10 15:22:06
import numpy as np
import pandas as pd
import re

import logging

import lib

from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, hashing_trick, text_to_word_sequence
from keras.utils import to_categorical

MAX_LABEL = 7 # Number of classes to predict. Update if number of case types changes.
MAX_WORDS = 400 # maximum number of words to keep, only most common MAX_WORDS will be kept

def generate_no_1095_train_test_data(pd_raw_data):
	'''
	Generates balanced Train and imbalanced Test data. 
	'''
	X = pd_raw_data['cog_Description']
	Y = pd_raw_data['cog_CaseTypeID']
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

	pd_imabalanced_train_data = pd.DataFrame()
	pd_imabalanced_train_data['cog_Description'] = X_train
	pd_imabalanced_train_data['cog_CaseTypeID'] = y_train

	pd_imabalanced_test_data = pd.DataFrame()
	pd_imabalanced_test_data['cog_Description'] = X_test
	pd_imabalanced_test_data['cog_CaseTypeID'] = y_test


	pass

def text_vectorization(text_data):
	'''
	Utility function to vectorize text data
	Different vectorization methods can be tested. Refer to README for suggestions.
	'''
	tokenizer = Tokenizer(num_words=MAX_WORDS)
	tokenizer.fit_on_texts(text_data)

	# TF-IDF (Term Frequency Inverse Document Frequency) method is used here.
	x = tokenizer.texts_to_matrix(text_data, 'tfidf')
	x = x.reshape(len(x), 1, MAX_WORDS)

	return x

def process_data(pd_data, train=False, evaluate=False):
	# Initialize logger
	LOG_FILENAME = 'case_automation.log'
	logging.basicConfig(filename=LOG_FILENAME,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

	logging.info('Processing data')

	text_data = pd_data['cog_Description'].values
	x = vectorization(text_data)

	if train:
		#pd_data = balance_raw_data(pd_data)
		label_data = pd_data['cog_CaseTypeID'].values - 1
		y_train = label_data.astype(int)

		# Convert label to categorical array: 5 = [0, 0, 0, 0, 1, 0, 0]
		categorial_y_train = to_categorical(y_train, num_classes=MAX_LABEL)


	if train:
		return x, categorial_y_train

	if evaluate:
		label_data = pd_data['cog_CaseTypeID'].values - 1
		y_test = label_data.astype(int)

		return x, y_test

	return x