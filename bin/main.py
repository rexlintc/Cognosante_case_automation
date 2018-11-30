# -*- coding: utf-8 -*-
# @Author: rlin
# @Date:   2018-07-30 17:36:07
# @Last Modified by:   rlin
# @Last Modified time: 2018-07-31 14:28:18

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import logging

import train
import evaluate
import predict

def main():

	logging.getLogger().setLevel(logging.INFO)

	if FLAGS.train:
		logging.INFO('Training a new model')
		train.

	if FLAGS.evaluate:
		logging.INFO('Evaluating the trained model')

	if FLAGS.predict:
    logging.INFO('Making predictions')

	pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train',
      default=False,
      help='Train the model.',
      action='store_true')
  parser.add_argument(
      '--evaluate',
      default=False,
      help='Evaluate the trained model.',
      action='store_true')
  parser.add_argument(
      '--predict',
      default=False,
      help='Predict.',
      action='store_true')
  FLAGS, unparsed = parser.parse_known_args()