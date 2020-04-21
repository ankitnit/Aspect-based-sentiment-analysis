#!/usr/bin/env python
# coding: utf-8


#Import library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging 
from sklearn.utils import shuffle
import string
import pickle
import tensorflow as tf
import os
from scipy import spatial
import argparse
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Embedding, Input, Dense, Activation, Conv1D, MaxPooling1D, Flatten, Concatenate, RepeatVector, dot, Reshape, Permute
from tensorflow.keras.models import Model, save_model,load_model
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# define constant


max_len = 64
pad_type = 'post'
trunc_type = 'post'
text_embedding = 128
embedding_dim = 300
oov_token = '<OOV>'
model_gigaword = api.load("glove-wiki-gigaword-300")



# read test data


def load_test_data(test_path):
    try:
        test_data = pd.read_excel(test_path)
    except:
        logger.error('File path is incorrect : {}'.format(test_path))
        raise
    return test_data


# text cleaning 

def clean(text):
    trastab = str.maketrans(string.punctuation,' '*len(string.punctuation))
    text= text.translate(trastab)
    text = text.lower()
    text = ' '.join([word for word in text.split()])
    return text

# vocab building

def vocab_(data):
    words = []
    for row in data.index:
        words.extend(data['Sentence'][row].split())
    return set(words)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# embedding vec generation 


def word_embedding(model_gigaword,word_index, dim = 300, ):
    embed_matrix = np.zeros(shape = (len(word_index)+1,dim))
    for word,index in word_index.items():
      if word in model_gigaword.vocab:
        embed_vec = model_gigaword[word]
        embed_matrix[index] = embed_vec
    print('Embedding completed')
    return embed_matrix




# find word similar to unseen word in existing vocab


def find_closest_embeddings(embedding,embeddings_matrix):
    dot = np.dot(embeddings_matrix, embedding)
    k  = np.argmax(dot)
    return k


# Index 2 word dictionary generation


def index2word(word2index):
    index_word = {}
    for key,value in word2index.items():
        index_word[value] = key
    return index_word


# test data preprocessing


def test_data_preprocess(test_data, oov_token, pad_type, trunc_type,word_gigaword, embedding_matrix, max_len=64, ):
    # loading
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    test_data['Sentence'] = test_data['Sentence'].apply(clean)
    vocab = vocab_(test_data)
    
    ##Word replacing
    word_index = tokenizer.word_index
    index_word = index2word(word_index)
    replace_dict = {}
    unseen_word = 0
    for key in vocab:
        if key not in list(word_index.keys()):
            # process a sentence using the model
            if key in word_gigaword.vocab:
                vec = word_gigaword[key]
                word = find_closest_embeddings(vec,embedding_matrix)
                replace_dict[key] = index_word[word]
                print('{} -----transformed to----{} '.format(key,index_word[word]))
                unseen_word += 1
                
    print('-----total unseen word transformed------{}'.format(unseen_word))
    test_data = test_data.replace(replace_dict) 
            
    ##test sentence preprocessing
    
    test_data['Preprocessed_Entity'] = test_data['Entity'].apply(lambda x: x + ' ' + x if len(x.split()) == 1 else x)
    test_sequence = tokenizer.texts_to_sequences(test_data['Sentence'])
    test_entity = tokenizer.texts_to_sequences(test_data['Preprocessed_Entity'])
    test_padded = pad_sequences(test_sequence, maxlen=max_len, padding=pad_type, truncating=trunc_type)
    test_entity_padded = pad_sequences(test_entity, maxlen=2)

    print('Shape of train data tensor:', test_padded.shape)


    return test_padded, test_entity_padded






if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    requiredNamed = ap.add_argument_group('required named arguments')
    requiredNamed.add_argument("-test_path", "--test_path", required=True,help="path to train data")
    requiredNamed.add_argument("-model_path", "--model_path", required=True,help="path to save the model")
    args = ap.parse_args()

    # Load test data

    test_data = load_test_data(args.test_path)
    logger.info('test data loded')

    
    # loading word_embed matrix for test time
    with open('embed_matrix.pickle', 'rb') as handle:
        embed_matrix = pickle.load(handle)
    logger.info('embedding matrix loded for finding similar word for unseen word in train vocab')

    #load trained model
    dependencies = {'f1_m' : f1_m, 'recall_m' : recall_m, 'precision_m' : precision_m }
    model = load_model(args.model_path, custom_objects = dependencies)
    logger.info('model loded')


    # test data preprocessing


    test_padded, test_entity_padded = test_data_preprocess(test_data, oov_token, pad_type, trunc_type, model_gigaword, embed_matrix)
    logger.info('Test data preprocessed')

    # Predict

    pred = model.predict([test_padded, test_entity_padded])
    logger.info('Result Generated')


    # Visualizing prediction


    pred_n = pred[pred>=0.5]
    pred_p = pred[pred<0.5]
    plt.scatter(pred_p,range(len(pred_p)),c = 'r')
    plt.scatter(pred_n,range(len(pred_n)),c = 'b')
    plt.show()
    pd.Series(pred[:,0]).plot(kind='hist')
    plt.show()
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0
    logger.info('1 : {}'.format(pred[pred==1].shape))
    logger.info('0 : {}'.format(pred[pred==0].shape))



    # Saving result
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0
    test_data['result'] = pred
    test_data['result'].replace({0 : 'Positive', 1 : 'Negative'}, inplace = True)
    test_data.to_excel('result.xlsx')
    logger.info('Result saved : result.xlsx')




