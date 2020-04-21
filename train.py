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


#word embedding

model_gigaword = api.load("glove-wiki-gigaword-300")


# define constant


max_len = 64
pad_type = 'post'
trunc_type = 'post'
text_embedding = 128
embedding_dim = 300
oov_token = '<OOV>'



# read train data


def load_train_data(train_path):
    try:
        train_data = pd.read_excel(train_path)
    except:
        logger.error('File path is incorrect : {}'.format(train_path))
        raise
    return train_data



# oversampling as data is imbalanced

def oversampling(data):
    pos_len = len(data[data['Sentiment'] == 'positive'])
    neg_index = data[data['Sentiment'] == 'negative'].index
    random_index = np.random.choice(neg_index, size = pos_len, replace=True, )
    pos_index = data[data['Sentiment'] == 'positive'].index
    index = np.concatenate([random_index, pos_index])
    train_data = data.loc[index]
    train_data = shuffle(train_data)
    train_data = train_data.reset_index(drop=True)
    return train_data

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


# train data preprocessing 


def preprocess(train_data, oov_token,pad_type,trunc_type, max_len=64,):
    ##preprocessing entity feature
    train_data['Sentence'] = train_data['Sentence'].apply(clean)
    vocab = len(vocab_(train_data))
    train_data['Preprocessed_Entity'] = train_data['Entity'].apply(lambda x: x + ' ' + x if len(x.split()) == 1 else x)

    ##vocab building, indexing
    tokenizer = Tokenizer(num_words = vocab, oov_token = oov_token)
    tokenizer.fit_on_texts(train_data['Sentence'])
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    ##train sentence preprocessing
    train_sequence = tokenizer.texts_to_sequences(train_data['Sentence'])
    train_entity = tokenizer.texts_to_sequences(train_data['Preprocessed_Entity'])
    train_padded = pad_sequences(train_sequence, maxlen=max_len, padding=pad_type, truncating=trunc_type)
    train_entity_padded = pad_sequences(train_entity, maxlen=2)


    ##label encoding
    '''
    Negative : 1
    Positive : 0
    '''
    label = train_data['Sentiment'].replace({"positive": 0, "negative": 1})

    # saving tokenizer class
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Shape of train data tensor:', train_padded.shape)
    print('Shape of train label tensor:', label.shape)

    return train_padded, train_entity_padded, label, word_index


# embedding vec generation 

def word_embedding(model_gigaword,word_index, dim = 300, ):
    embed_matrix = np.zeros(shape = (len(word_index)+1,dim))
    for word,index in word_index.items():
      if word in model_gigaword.vocab:
        embed_vec = model_gigaword[word]
        embed_matrix[index] = embed_vec
    print('Embedding completed')
    return embed_matrix




# calculate precision, recall, and f1 score



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


# Model architecture


def build_model(embedding_layer,embedding_layer_entity,max_len):
    sequence_input = Input(shape=(max_len,))
    entity_input = Input(shape=(2,),)
    embedded_sequences = embedding_layer(sequence_input)
    embedded_entity = embedding_layer_entity(entity_input)
    #print(entity_input.shape)
    x = Conv1D(128, 3, activation='relu',padding='same')(embedded_sequences)
    x1 = Conv1D(128, 2, activation='relu')(embedded_entity)
    ###aspect based attention block
    con = Concatenate(axis = 1)([x,x1])
    x2 = Dense(1,activation= 'tanh')(con)
    x2 = Flatten()(x2)
    x2 = Activation('softmax')(x2)
    x2 = RepeatVector(64)(x2)
    x2 = dot([x,x2],axes = 1)
    x2 = Permute([2, 1])(x2)
    ###attention end
    x = MaxPooling1D(3)(x2)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(3)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(3)(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    #x = concatenate([x,d])
    preds = Dense(1, activation='sigmoid')(x)

    model = Model([sequence_input,entity_input], preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', 
                  metrics=['acc',f1_m,precision_m, recall_m])
    return model


# training model without kfold cv


def train(word_index, embed_matrix, max_len, train_padded, train_entity_padded, label, epoch = 15, batch_size = 128):
    ##sentence embedding layer
    embedding_layer = Embedding(input_dim=len(word_index) + 1, output_dim=300,
                                weights=[embed_matrix], input_length=max_len, trainable=True)

    ##entity embedding layer
    embedding_layer_entity = Embedding(input_dim=len(word_index) + 1, output_dim=300,
                                       weights=[embed_matrix], input_length=2, trainable=True)


    ##sentence, entity and label
    x = train_padded
    y = label
    x_e = train_entity_padded

    ##label preprocessing
    y_ = np.asarray(y, dtype='int')

    model = build_model(embedding_layer, embedding_layer_entity, max_len)


    ##
    logger.info('Training started')
    history = model.fit([x, x_e], y_, epochs = epoch, batch_size = batch_size)
    logger.info('Training completed')
    return model




# Training model in kfold CV


def train_kfold(word_index, embed_matrix, max_len, train_padded, train_entity_padded, kfold, label, epoch = 15, batch_size = 128):
    ##sentence embedding layer
    embedding_layer = Embedding(input_dim=len(word_index) + 1, output_dim=300,
                                weights=[embed_matrix], input_length=max_len, trainable=False)

    ##entity embedding layer
    embedding_layer_entity = Embedding(input_dim=len(word_index) + 1, output_dim=300,
                                       weights=[embed_matrix], input_length=2, trainable=False)

    ##kfold cross validation
    skf = StratifiedKFold(n_splits = kfold, )
    score = pd.DataFrame(columns=['loss', 'acc', 'rec', 'pre', 'f1'])
    index = 0

    ##sentence, entity and label
    x = train_padded
    y = label
    x_e = train_entity_padded

    ##label preprocessing
    y_ = np.asarray(y, dtype='int')

    ##kfold cross validation
    '''
    Result will be stored in Pandas dataframe with name as Score 
    
    '''
    logger.info('Result will be stored in Pandas dataframe with name as Score ')
    logger.info('Starting K-Fold cross validation')
    for train, test in skf.split(x, y_):
        model = build_model(embedding_layer, embedding_layer_entity, max_len)

        ##
        xtrain, xetrain, ytrain = x[train], x_e[train], y_[train]
        xtest, xetest, ytest = x[test], x_e[test], y_[test]

        ##
        logger.info('Training Strated for {} of {} fold'.format(index, kfold))
        history = model.fit([xtrain, xetrain], ytrain, epochs = epoch, batch_size = batch_size)

        ##################
        pred = model.predict([xtest, xetest])
        pred_neg = pred[pred >= 0.5]
        pred_pos = pred[pred < 0.5]
        plt.scatter(pred_pos, range(len(pred_pos)), c='r')
        plt.scatter(pred_neg, range(len(pred_neg)), c='b')
        plt.ylabel('Prob')
        plt.xlabel('Epoch')
        plt.legend(['Positive','Negative'], loc='upper left')
        plt.show()
        pd.Series(pred[:, 0]).plot(kind='hist')
        plt.ylabel('prob')
        plt.xlabel('Epoch')
        plt.show()
        ######################

        val = model.evaluate([xtest, xetest], ytest, verbose=0)
        score.loc[index] = val
        index += 1
        print('score : ', score)
        #################3
        plt.plot(history.history['acc'], 'r')
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
    logger.info('trainig completed')
    return score



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--batch_size", default = 128, required=False,help="batch_size")
    ap.add_argument("-d", "--dim", default = 300, required=False,help="embedding dimension")
    ap.add_argument("-kf", "--kfold", default = 5, required=False,help="number of fold")
    ap.add_argument("-i", "--epoch", default = 15, required=False,help="number of epochs")
    requiredNamed = ap.add_argument_group('required named arguments')
    requiredNamed.add_argument("-iskf", "--iskfold",required=False,help="allow to train as  kfold CV --give [True, False]")
    requiredNamed.add_argument("-train_path", "--train_path", required=True,help="path to train data")
    requiredNamed.add_argument("-model_path", "--model_path", required=True,help="path to save the model")
    args = ap.parse_args()

    # Load train and test data

    train_data = load_train_data(args.train_path)
    logger.info('train data loded')

    # generate oversampled train data

    train_data = oversampling(train_data)
    logger.info('oversampling done')

    # Preprocess the train data

    train_padded, train_entity_padded, label, word_index = preprocess(train_data, oov_token,pad_type,trunc_type, max_len=64,)
    logger.info('Text preprocessing done')


    # Generate embedding
    embed_matrix = word_embedding(model_gigaword, word_index,)
    # saving word_embed matrix for test time
    with open('embed_matrix.pickle', 'wb') as handle:
        pickle.dump(embed_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)


    #Training started for Kfold
    logger.info('Training started for Kfold') 
    if args.iskfold == 'True':
        logger.info('Kfold CV ')
        score = train_kfold(word_index, embed_matrix, max_len, train_padded, train_entity_padded, args.kfold, label, epoch = int(args.epoch), batch_size = int(args.batch_size))
        logger.info('view precision, recall f1, loss and accuracy score')
        logger.info('{}'.format(score))
        logger.info('Training completed for Kfold')


    # Training without Kfold
    logger.info('Training started for without Kfold')
    model = train(word_index, embed_matrix, max_len, train_padded, train_entity_padded, label, epoch = int(args.epoch), batch_size = int(args.batch_size))
    logger.info('Training completed without Kfold')
    model.save('model.h5')
    logger.info('model saved')



