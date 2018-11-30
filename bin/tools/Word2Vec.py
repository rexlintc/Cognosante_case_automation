
# coding: utf-8

import numpy as np
import pandas as pd
import logging

from gensim.models import word2vec

import nltk
# Run this the first time for installation
# nltk.download('all')
from nltk.tokenize import sent_tokenize # tokenizes sentences
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

eng_stopwords = stopwords.words('english')


def narrative_cleaner(narrative, lemmatize=True, stem=False):
    '''
    Clean and preprocess a narrative.

    1. Use regex to remove all special characters (only keep letters)
    2. Make strings to lower case and tokenize / word split reviews
    3. Remove English stopwords
    4. Rejoin to one string
    '''
    ps = PorterStemmer()
    wnl = WordNetLemmatizer()
    
    #1. Remove punctuation
    narrative = re.sub("[^a-zA-Z]", " ", narrative)
    
    #2. Tokenize into words (all lower case)
    narrative = narrative.lower().split()
    
    #3.Set stopwords
    eng_stopwords = set(stopwords.words("english"))

    clean_narrative = []
    for word in narrative:
        if word not in eng_stopwords:
            if lemmatize is True:
                word = wnl.lemmatize(word)
            elif stem is True:
                if word == 'oed':
                    continue
                word = ps.stem(word)
            clean_narrative.append(word)
    return(clean_narrative)


def build_save_w2v(narratives, num_features=300, min_word_count=1, num_workers=2, context=10, save=True, save_directory):
    """
    Builds and saves Word2Vec Model
    """
    # Preprocess narratives
    processed_narratives = []
    for i in range(len(narratives)):
        if((i + 1) % 5000 == 0 ):
            # print progress
            print("Done with {} reviews".format(i+1)) 
        processed_narratives.append(narrative_cleaner(narratives[i]))
    
    sentences = processed_narratives
    
    # Set values for various parameters
    num_features = num_features    # Word vector dimensionality                      
    min_word_count = min_word_count   # ignore all words with total frequency lower than this                       
    num_workers = num_workers       # Number of threads to run in parallel
    context = context        # Context window size 
    
    logging.info("Training word2vec model... ")
    model = word2vec.Word2Vec(sentences, workers=num_workers,                size=num_features, min_count = min_word_count,                 window = context)


    if save:
        # save the model for later use
        model_name = "{}features_{}minwords_{}context".format(num_features, min_word_count, context)
        model.save(model_name)